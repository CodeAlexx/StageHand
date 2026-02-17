"""Block registry for the stagehand runtime.

Maps model topology to an ordered set of block entries used by the
scheduler, residency map, and transfer engine.  Built once at model load
time by walking ``model.named_modules()`` and matching against a caller-
supplied pattern.  Immutable after construction + validation.

First target model: WAN 2.2 (video diffusion with temporal + spatial
attention blocks).
"""
from __future__ import annotations

import json
import re
import struct
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from stagehand.errors import StagehandOOMError

if TYPE_CHECKING:
    pass

__all__ = [
    "BlockEntry",
    "BlockRegistry",
    "FileParamSpec",
    "SquareQParamSpec",
    "ParamLayoutEntry",
]


ParamLayoutEntry = tuple[str, tuple[int, ...], torch.dtype, int, int]


@dataclass(frozen=True)
class FileParamSpec:
    """Descriptor for a parameter sourced from safetensors on disk."""

    param_name: str
    file_offset: int
    source_nbytes: int
    source_dtype: torch.dtype
    source_shape: tuple[int, ...]
    file_path: str = ""  # Per-param file path for sharded models (empty = use block source_path)


@dataclass(frozen=True)
class SquareQParamSpec:
    """Descriptor for a parameter sourced from a SquareQ BP8 slab."""

    param_name: str
    layer_name: str
    kind: str  # "weight" or "bias"
    out_features: int
    in_features: int
    padded_in_features: int


# ── helpers ──────────────────────────────────────────────────────────────


def _param_size_bytes(module: nn.Module, dtype: torch.dtype) -> int:
    """Sum of all parameter sizes in *module* when stored as *dtype*."""
    total = 0
    for p in module.parameters():
        total += p.numel() * dtype.itemsize
    return total


# ── data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockEntry:
    """Immutable descriptor for a single swappable block.

    Mirrors spec Section 2.2.1.  ``module_ref`` is a weak reference so the
    registry never prevents garbage collection of the underlying module.
    """

    block_id: str
    module_ref: weakref.ref  # weak reference to nn.Module
    size_bytes: int
    dtype: torch.dtype
    dependencies: tuple[str, ...]
    group: str
    exec_order: int
    quant_format: str | None = None
    quant_meta_bytes: int = 0
    # File-backed mode metadata. Empty for module-backed blocks.
    source_path: str | None = None
    source_format: str | None = None
    param_layout: tuple[ParamLayoutEntry, ...] = field(default_factory=tuple)
    file_param_specs: tuple[FileParamSpec, ...] = field(default_factory=tuple)
    squareq_param_specs: tuple[SquareQParamSpec, ...] = field(default_factory=tuple)
    module_param_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def file_backed(self) -> bool:
        return self.source_path is not None and (
            len(self.file_param_specs) > 0 or len(self.squareq_param_specs) > 0
        )

    @property
    def squareq_backed(self) -> bool:
        return self.source_format == "squareq_bp8" and len(self.squareq_param_specs) > 0


# ── registry ─────────────────────────────────────────────────────────────


class BlockRegistry:
    """Ordered, immutable registry of swappable blocks.

    Typical usage::

        registry = BlockRegistry()
        registry.build_from_model(wan_model, block_pattern="spatial|temporal", group="wan", dtype=torch.bfloat16)
        registry.validate(pool_capacity_bytes=8 * 1024**3)
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, BlockEntry] = OrderedDict()
        self._frozen: bool = False

    @staticmethod
    def _decode_safetensors_dtype(dtype_name: str) -> torch.dtype:
        mapping: dict[str, torch.dtype] = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U8": torch.uint8,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported safetensors dtype {dtype_name!r}")
        return mapping[dtype_name]

    @staticmethod
    def _parse_safetensors_index(
        safetensors_path: Path,
    ) -> dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]]:
        """Return tensor index map for a safetensors file.

        Values are ``(abs_offset, nbytes, source_dtype, source_shape)``.
        """
        with safetensors_path.open("rb") as handle:
            header_len_raw = handle.read(8)
            if len(header_len_raw) != 8:
                raise ValueError(f"Invalid safetensors header for {safetensors_path}")
            header_len = struct.unpack("<Q", header_len_raw)[0]
            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError(f"Truncated safetensors header for {safetensors_path}")

        header = json.loads(header_bytes.decode("utf-8"))
        data_base = 8 + header_len
        index: dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]] = {}
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict):
                continue
            offsets = value.get("data_offsets")
            shape = value.get("shape")
            dtype_name = value.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not isinstance(shape, list)
                or not isinstance(dtype_name, str)
            ):
                continue
            start_rel, end_rel = int(offsets[0]), int(offsets[1])
            if end_rel < start_rel:
                raise ValueError(f"Invalid data_offsets for key {key!r} in {safetensors_path}")
            source_dtype = BlockRegistry._decode_safetensors_dtype(dtype_name)
            source_shape = tuple(int(dim) for dim in shape)
            index[key] = (
                data_base + start_rel,
                end_rel - start_rel,
                source_dtype,
                source_shape,
            )
        return index

    @staticmethod
    def _candidate_tensor_keys(block_id: str, param_name: str) -> tuple[str, ...]:
        """Return likely safetensors keys for a module parameter name.

        Adapter wrappers may rename base parameters (for example
        ``to_q.orig.weight``). The checkpoint keeps the original key
        (``to_q.weight``). This helper generates normalized candidates.
        """
        names: list[str] = []
        seen: set[str] = set()
        queue: list[str] = [param_name]
        tokens = (".orig.", ".org_module.", ".base_layer.")
        aliases = (
            ("attn1.", "self_attn."),
            ("attn2.", "cross_attn."),
            ("to_q.", "q."),
            ("to_k.", "k."),
            ("to_v.", "v."),
            ("to_out.0.", "o."),
            ("ffn.net.0.proj.", "ffn.0."),
            ("ffn.net.2.", "ffn.2."),
            ("norm2.", "norm3."),
            (".attn1.", ".self_attn."),
            (".attn2.", ".cross_attn."),
            (".to_q.", ".q."),
            (".to_k.", ".k."),
            (".to_v.", ".v."),
            (".to_out.0.", ".o."),
            (".ffn.net.0.proj.", ".ffn.0."),
            (".ffn.net.2.", ".ffn.2."),
            (".norm2.", ".norm3."),
            ("scale_shift_table", "modulation"),
        )

        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            names.append(current)

            for token in tokens:
                if token not in current:
                    continue
                normalized = current.replace(token, ".")
                normalized = normalized.replace("..", ".").strip(".")
                if normalized and normalized not in seen:
                    queue.append(normalized)

            for src, dst in aliases:
                if src not in current:
                    continue
                alias = current.replace(src, dst)
                alias = alias.replace("..", ".").strip(".")
                if alias and alias not in seen:
                    queue.append(alias)

        return tuple(f"{block_id}.{name}" for name in names)

    @staticmethod
    def _candidate_squareq_layer_keys(block_id: str, param_name: str) -> tuple[str, ...]:
        """Return likely SquareQ layer names for a module parameter."""
        base_name = param_name
        if base_name.endswith(".weight"):
            base_name = base_name[:-7]
        elif base_name.endswith(".bias"):
            base_name = base_name[:-5]
        elif base_name in {"weight", "bias"}:
            base_name = ""
        if not base_name:
            return (block_id,)
        return BlockRegistry._candidate_tensor_keys(block_id, base_name)

    @staticmethod
    def _drop_param_storage(module: nn.Module, param_name: str, param: nn.Parameter) -> None:
        """Release parameter storage while preserving module parameter wiring.

        Torch rejects ``param.data = cpu_tensor`` when the existing parameter is
        CUDA-backed or meta-backed. In those cases we replace the Parameter
        object at its owning module path with an empty CPU parameter.
        """
        empty = torch.empty(0, dtype=param.dtype, device="cpu")
        with torch.no_grad():
            if param.device.type == "cpu" and not param.is_meta:
                param.data = empty
                if param.grad is not None:
                    param.grad = None
                return

            owner: nn.Module = module
            parts = param_name.split(".")
            for part in parts[:-1]:
                owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
            leaf = parts[-1]
            replacement = nn.Parameter(empty, requires_grad=param.requires_grad)
            if hasattr(owner, "_parameters") and leaf in owner._parameters:
                owner._parameters[leaf] = replacement
            else:
                setattr(owner, leaf, replacement)

    @staticmethod
    def _load_squareq_manifest(
        squareq_path: Path,
    ) -> dict[str, tuple[int, int, int, bool]]:
        """Load SquareQ slab manifest metadata.

        Returns a mapping:
            layer_name -> (out_features, in_features, padded_in_features, has_bias)
        """
        payload = None
        for kwargs in (
            {"weights_only": True, "mmap": True},
            {"weights_only": True},
        ):
            try:
                payload = torch.load(squareq_path, map_location="cpu", **kwargs)
                break
            except TypeError:
                continue

        if payload is None:
            payload = torch.load(squareq_path, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid SquareQ slab payload in {squareq_path}")
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"SquareQ slab missing manifest in {squareq_path}")

        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise ValueError(f"SquareQ manifest missing layers list in {squareq_path}")

        result: dict[str, tuple[int, int, int, bool]] = {}
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            name = layer.get("name")
            if not isinstance(name, str) or not name:
                continue
            out_features = int(layer.get("out", 0))
            in_features = int(layer.get("inp", 0))
            padded_in = int(layer.get("padded_in", in_features))
            has_bias = bool(layer.get("has_bias", False))
            if out_features <= 0 or in_features <= 0:
                continue
            result[name] = (out_features, in_features, padded_in, has_bias)

        return result

    # ── construction ─────────────────────────────────────────────────

    def build_from_model(
        self,
        model: nn.Module,
        block_pattern: str,
        group: str,
        dtype: torch.dtype,
    ) -> None:
        """Walk *model* and register every module whose name matches *block_pattern*.

        Parameters
        ----------
        model:
            The PyTorch model to scan.
        block_pattern:
            Regex pattern matched against the fully-qualified module name
            (as returned by ``named_modules()``).
        group:
            Logical group label (e.g. ``"wan"``, ``"dit"``, ``"te1"``).
        dtype:
            Target dtype used to compute ``size_bytes``.

        Raises
        ------
        RuntimeError
            If the registry has already been frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot build_from_model on a frozen registry")

        compiled = re.compile(block_pattern)
        order = len(self._entries)

        for name, module in model.named_modules():
            if not name:
                continue
            if compiled.search(name):
                block_id = name
                size = _param_size_bytes(module, dtype)
                entry = BlockEntry(
                    block_id=block_id,
                    module_ref=weakref.ref(module),
                    size_bytes=size,
                    dtype=dtype,
                    dependencies=(),
                    group=group,
                    exec_order=order,
                )
                self._entries[block_id] = entry
                order += 1

    def validate(self, pool_capacity_bytes: int) -> None:
        """Check all blocks fit in the pool and freeze the registry.

        Raises
        ------
        StagehandOOMError
            If any single block exceeds *pool_capacity_bytes*.
        """
        for entry in self._entries.values():
            if entry.size_bytes > pool_capacity_bytes:
                raise StagehandOOMError(
                    f"Block {entry.block_id!r} ({entry.size_bytes} bytes) "
                    f"exceeds pool capacity ({pool_capacity_bytes} bytes)"
                )
        self._frozen = True

    def convert_to_file_backed(
        self,
        source_path: str | Path,
        *,
        drop_module_tensors: bool = True,
    ) -> int:
        """Convert eligible block params to file-backed descriptors.

        This keeps the module scaffold (for hooks / topology) while dropping
        frozen parameter tensors from CPU RAM.

        Returns
        -------
        int
            Number of parameters converted to file-backed references.
        """
        if not self._frozen:
            raise RuntimeError("convert_to_file_backed requires a validated (frozen) registry")

        path = Path(source_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"file-backed source not found: {path}")

        if path.suffix.lower() in {".fpk", ".slab"}:
            return self.convert_to_squareq_backed(path, drop_module_tensors=drop_module_tensors)

        tensor_index = self._parse_safetensors_index(path)
        converted_params = 0
        new_entries: OrderedDict[str, BlockEntry] = OrderedDict()

        for block_id, entry in self._entries.items():
            module = entry.module_ref()
            if module is None:
                new_entries[block_id] = entry
                continue

            layout: list[ParamLayoutEntry] = []
            file_specs: list[FileParamSpec] = []
            module_param_names: list[str] = []
            offset = 0

            for param_name, param in module.named_parameters():
                shape = tuple(int(dim) for dim in param.shape)
                numel = int(param.numel())
                nbytes = numel * entry.dtype.itemsize
                layout.append((param_name, shape, entry.dtype, offset, numel))
                offset += nbytes

                if param.requires_grad:
                    module_param_names.append(param_name)
                    continue

                tensor_meta = None
                for tensor_key in self._candidate_tensor_keys(block_id, param_name):
                    tensor_meta = tensor_index.get(tensor_key)
                    if tensor_meta is not None:
                        break
                if tensor_meta is None:
                    module_param_names.append(param_name)
                    continue

                file_offset, source_nbytes, source_dtype, source_shape = tensor_meta
                if source_shape != shape:
                    module_param_names.append(param_name)
                    continue

                file_specs.append(
                    FileParamSpec(
                        param_name=param_name,
                        file_offset=file_offset,
                        source_nbytes=source_nbytes,
                        source_dtype=source_dtype,
                        source_shape=source_shape,
                    )
                )
                converted_params += 1

                if drop_module_tensors:
                    self._drop_param_storage(module, param_name, param)

            if file_specs:
                new_entry = replace(
                    entry,
                    source_path=str(path),
                    source_format="safetensors",
                    param_layout=tuple(layout),
                    file_param_specs=tuple(file_specs),
                    squareq_param_specs=tuple(),
                    module_param_names=tuple(module_param_names),
                )
                new_entries[block_id] = new_entry
            else:
                new_entries[block_id] = entry

        self._entries = new_entries
        return converted_params

    def convert_to_file_backed_sharded(
        self,
        source_dir: str | Path,
        *,
        drop_module_tensors: bool = True,
    ) -> int:
        """Convert eligible block params to file-backed descriptors from sharded safetensors.

        Reads ``diffusion_pytorch_model.safetensors.index.json`` to find the
        tensor→shard mapping, then parses each shard's header to build a
        unified tensor index with per-param file paths.

        Parameters
        ----------
        source_dir:
            Directory containing sharded safetensors files and their index JSON.
        drop_module_tensors:
            If *True*, replace frozen parameter data with empty tensors to free
            CPU RAM after conversion.

        Returns
        -------
        int
            Number of parameters converted to file-backed references.
        """
        if not self._frozen:
            raise RuntimeError("convert_to_file_backed_sharded requires a validated (frozen) registry")

        dirpath = Path(source_dir).expanduser()
        if not dirpath.is_dir():
            raise FileNotFoundError(f"Sharded source directory not found: {dirpath}")

        # Find and load the shard index JSON
        index_path = dirpath / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.exists():
            # Try model.safetensors.index.json as fallback
            index_path = dirpath / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No safetensors index found in {dirpath}. "
                "Expected diffusion_pytorch_model.safetensors.index.json"
            )

        with index_path.open("r") as f:
            shard_index = json.loads(f.read())

        weight_map: dict[str, str] = shard_index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Empty weight_map in {index_path}")

        # Group tensors by shard file, then parse each shard's header once
        shard_to_keys: dict[str, list[str]] = {}
        for tensor_key, shard_file in weight_map.items():
            shard_to_keys.setdefault(shard_file, []).append(tensor_key)

        # Build unified tensor index: key -> (abs_offset, nbytes, dtype, shape, shard_abs_path)
        unified_index: dict[str, tuple[int, int, torch.dtype, tuple[int, ...], str]] = {}
        for shard_file in shard_to_keys:
            shard_path = dirpath / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            shard_tensor_index = self._parse_safetensors_index(shard_path)
            abs_shard_path = str(shard_path)
            for key, (offset, nbytes, dtype, shape) in shard_tensor_index.items():
                unified_index[key] = (offset, nbytes, dtype, shape, abs_shard_path)

        # Now convert blocks using the unified index
        converted_params = 0
        new_entries: OrderedDict[str, BlockEntry] = OrderedDict()

        for block_id, entry in self._entries.items():
            module = entry.module_ref()
            if module is None:
                new_entries[block_id] = entry
                continue

            layout: list[ParamLayoutEntry] = []
            file_specs: list[FileParamSpec] = []
            module_param_names: list[str] = []
            offset = 0

            for param_name, param in module.named_parameters():
                shape = tuple(int(dim) for dim in param.shape)
                numel = int(param.numel())
                nbytes = numel * entry.dtype.itemsize
                layout.append((param_name, shape, entry.dtype, offset, numel))
                offset += nbytes

                if param.requires_grad:
                    module_param_names.append(param_name)
                    continue

                tensor_meta = None
                for tensor_key in self._candidate_tensor_keys(block_id, param_name):
                    tensor_meta = unified_index.get(tensor_key)
                    if tensor_meta is not None:
                        break
                if tensor_meta is None:
                    module_param_names.append(param_name)
                    continue

                file_offset, source_nbytes, source_dtype, source_shape, shard_path = tensor_meta
                if source_shape != shape:
                    module_param_names.append(param_name)
                    continue

                file_specs.append(
                    FileParamSpec(
                        param_name=param_name,
                        file_offset=file_offset,
                        source_nbytes=source_nbytes,
                        source_dtype=source_dtype,
                        source_shape=source_shape,
                        file_path=shard_path,
                    )
                )
                converted_params += 1

                if drop_module_tensors:
                    self._drop_param_storage(module, param_name, param)

            if file_specs:
                new_entry = replace(
                    entry,
                    source_path=str(dirpath),
                    source_format="safetensors_sharded",
                    param_layout=tuple(layout),
                    file_param_specs=tuple(file_specs),
                    squareq_param_specs=tuple(),
                    module_param_names=tuple(module_param_names),
                )
                new_entries[block_id] = new_entry
            else:
                new_entries[block_id] = entry

        self._entries = new_entries
        return converted_params

    def convert_to_squareq_backed(
        self,
        squareq_path: str | Path,
        *,
        drop_module_tensors: bool = True,
    ) -> int:
        """Convert eligible block params to SquareQ BP8-backed descriptors."""
        if not self._frozen:
            raise RuntimeError("convert_to_squareq_backed requires a validated (frozen) registry")

        path = Path(squareq_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SquareQ slab file not found: {path}")

        layer_meta = self._load_squareq_manifest(path)
        converted_params = 0
        new_entries: OrderedDict[str, BlockEntry] = OrderedDict()

        for block_id, entry in self._entries.items():
            module = entry.module_ref()
            if module is None:
                new_entries[block_id] = entry
                continue

            layout: list[ParamLayoutEntry] = []
            squareq_specs: list[SquareQParamSpec] = []
            module_param_names: list[str] = []
            offset = 0

            for param_name, param in module.named_parameters():
                shape = tuple(int(dim) for dim in param.shape)
                numel = int(param.numel())
                nbytes = numel * entry.dtype.itemsize
                layout.append((param_name, shape, entry.dtype, offset, numel))
                offset += nbytes

                if param.requires_grad:
                    module_param_names.append(param_name)
                    continue

                kind: str | None = None
                if param_name.endswith(".weight") or param_name == "weight":
                    kind = "weight"
                elif param_name.endswith(".bias") or param_name == "bias":
                    kind = "bias"
                if kind is None:
                    module_param_names.append(param_name)
                    continue

                matched_layer: str | None = None
                matched_meta: tuple[int, int, int, bool] | None = None
                for layer_name in self._candidate_squareq_layer_keys(block_id, param_name):
                    meta = layer_meta.get(layer_name)
                    if meta is not None:
                        matched_layer = layer_name
                        matched_meta = meta
                        break
                if matched_layer is None or matched_meta is None:
                    module_param_names.append(param_name)
                    continue

                out_features, in_features, padded_in_features, has_bias = matched_meta
                if kind == "weight":
                    if len(shape) == 0:
                        module_param_names.append(param_name)
                        continue
                    out_dim = int(shape[0])
                    in_flat = int(numel // max(out_dim, 1))
                    if (
                        out_dim != out_features
                        or in_flat != in_features
                        or out_dim <= 0
                        or in_flat <= 0
                    ):
                        module_param_names.append(param_name)
                        continue
                else:
                    if not has_bias or numel != out_features:
                        module_param_names.append(param_name)
                        continue

                squareq_specs.append(
                    SquareQParamSpec(
                        param_name=param_name,
                        layer_name=matched_layer,
                        kind=kind,
                        out_features=out_features,
                        in_features=in_features,
                        padded_in_features=padded_in_features,
                    )
                )
                converted_params += 1

                if drop_module_tensors:
                    self._drop_param_storage(module, param_name, param)

            if squareq_specs:
                new_entry = replace(
                    entry,
                    source_path=str(path),
                    source_format="squareq_bp8",
                    param_layout=tuple(layout),
                    file_param_specs=tuple(),
                    squareq_param_specs=tuple(squareq_specs),
                    module_param_names=tuple(module_param_names),
                )
                new_entries[block_id] = new_entry
            else:
                new_entries[block_id] = entry

        self._entries = new_entries
        return converted_params

    def build_from_module_list(
        self,
        modules: list[tuple[str, nn.Module]],
        group: str,
        dtype: torch.dtype,
    ) -> None:
        """Register an ordered list of ``(name, module)`` pairs.

        Used by layer mode where modules are discovered by walking
        ``named_modules()`` rather than matching a regex pattern.
        ``exec_order`` is assigned from list position.  Non-breaking —
        block-mode callers are unaffected.
        """
        if self._frozen:
            raise RuntimeError("Cannot build_from_module_list on a frozen registry")

        seen_ids: set[int] = set()
        order = len(self._entries)

        for name, module in modules:
            mid = id(module)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            size = _param_size_bytes(module, dtype)
            entry = BlockEntry(
                block_id=name,
                module_ref=weakref.ref(module),
                size_bytes=size,
                dtype=dtype,
                dependencies=(),
                group=group,
                exec_order=order,
            )
            self._entries[name] = entry
            order += 1

    # ── queries ──────────────────────────────────────────────────────

    def get(self, block_id: str) -> BlockEntry:
        """Return the entry for *block_id*.

        Raises
        ------
        KeyError
            If *block_id* is not in the registry.
        """
        return self._entries[block_id]

    def blocks_in_order(self) -> list[BlockEntry]:
        """All entries sorted by ``exec_order``."""
        return sorted(self._entries.values(), key=lambda e: e.exec_order)

    def __len__(self) -> int:
        return len(self._entries)

    def groups(self) -> dict[str, list[BlockEntry]]:
        """Entries grouped by their ``group`` field."""
        result: dict[str, list[BlockEntry]] = {}
        for entry in self._entries.values():
            result.setdefault(entry.group, []).append(entry)
        return result

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._entries

    def __repr__(self) -> str:
        return f"BlockRegistry(blocks={len(self)}, frozen={self._frozen})"
