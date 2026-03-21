"""Stagehand scheduler and static lookahead eviction policy.

Orchestrates prefetching and eviction of transformer blocks between
host pinned memory and GPU, using the transfer engine, residency map,
budget manager, and telemetry components.

The :class:`StaticLookaheadPolicy` is the first (deterministic) policy
implementation — prefetch a fixed window ahead and evict by distance * size.
"""
from __future__ import annotations

import mmap
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from stagehand.residency import BlockState

if TYPE_CHECKING:
    from stagehand.config import StagehandConfig
    from stagehand.pool import PinnedPool, PinnedSlab
    from stagehand.registry import BlockEntry, BlockRegistry, FileParamSpec, SquareQParamSpec
    from stagehand.residency import ResidencyEntry, ResidencyMap
    from stagehand.telemetry import StagehandTelemetry
    from stagehand.transfer import AsyncTransferEngine, TransferHandle
    try:
        from serenity.memory.adapters.stagehand_conductor import StagehandConductorAdapter
        from serenity.memory.adapters.squareq_conductor import SquareQConductorAdapter
    except ImportError:
        pass
    try:
        from eriquant.stagehand.adapter import StagehandAdapter as EriQuantAdapter
    except ImportError:
        pass

__all__ = ["StaticLookaheadPolicy", "StagehandScheduler"]

log = logging.getLogger(__name__)
_STALL_LOGS_ENABLED = str(os.getenv("SERENITY_STAGEHAND_STALL_LOGS", "0")).strip().lower() in {"1", "true", "yes", "on"}


# ── protocols for optional components ────────────────────────────────────
# budget.py and guards.py may be built by another agent in parallel.
# We define lightweight Protocol interfaces here so the scheduler
# compiles regardless of their availability.


class BudgetLike(Protocol):
    """Minimal interface expected from BudgetManager."""

    def above_high_watermark(self) -> bool: ...
    def below_low_watermark(self) -> bool: ...


class GuardsLike(Protocol):
    """Minimal interface expected from NumericGuard."""

    def check_output(
        self, tensor: torch.Tensor, block_id: str, step: int,
    ) -> tuple[int, int]: ...


# ── static lookahead policy ──────────────────────────────────────────────


class StaticLookaheadPolicy:
    """Deterministic prefetch-ahead / eviction-scoring policy.

    Parameters
    ----------
    prefetch_window:
        Number of blocks ahead of the cursor to prefetch.
    eviction_cooldown_steps:
        Minimum steps since last use before a block may be evicted.
    """

    def __init__(
        self,
        prefetch_window: int = 3,
        eviction_cooldown_steps: int = 2,
    ) -> None:
        self.prefetch_window = prefetch_window
        self.eviction_cooldown_steps = eviction_cooldown_steps

    def blocks_to_prefetch(
        self,
        cursor: int,
        total_blocks: int,
        residency_states: dict[str, str],
    ) -> list[int]:
        """Return exec_order indices to prefetch.

        Looks from ``cursor + 1`` to ``cursor + prefetch_window`` (inclusive),
        skipping any that are already GPU_READY or PREFETCHING.

        Parameters
        ----------
        cursor:
            Current position in exec_order (0-based).
        total_blocks:
            Total number of blocks in the registry.
        residency_states:
            Mapping of block_id to state string for each block.
            Only used to filter out blocks already on GPU / in transit.
            Keys are block_ids; the caller maps exec_order indices to ids.

        Returns
        -------
        list[int]
            Exec_order indices that need prefetching, in order.
        """
        result: list[int] = []
        for offset in range(1, self.prefetch_window + 1):
            idx = cursor + offset
            if idx >= total_blocks:
                break
            result.append(idx)
        return result

    def score_for_eviction(
        self,
        block_id: str,
        current_cursor: int,
        entry_exec_order: int,
        total_blocks: int,
        size_bytes: int,
    ) -> float:
        """Score a block for eviction — higher score = better candidate.

        Formula: ``next_use_distance * size_bytes``

        ``next_use_distance`` wraps around for the next epoch.
        """
        next_use_distance = (entry_exec_order - current_cursor) % total_blocks
        if next_use_distance == 0:
            # Block is at cursor — it was just used or is about to be used.
            # Set distance to total_blocks (farthest away in wrap-around).
            next_use_distance = total_blocks
        return float(next_use_distance * size_bytes)

    def should_evict(self, last_used_step: int, current_step: int) -> bool:
        """True if enough steps have passed since the block was last used."""
        return current_step - last_used_step > self.eviction_cooldown_steps


# ── parameter layout helpers ─────────────────────────────────────────────


def _build_param_layout(
    module: nn.Module,
    dtype: torch.dtype,
) -> list[tuple[str, tuple[int, ...], torch.dtype, int, int]]:
    """Build a contiguous parameter layout for *module*.

    Returns a list of ``(param_name, shape, dtype, offset_bytes, num_elements)``
    tuples describing how each parameter is packed into a flat buffer.
    """
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]] = []
    offset = 0
    for name, param in module.named_parameters():
        numel = param.numel()
        elem_size = dtype.itemsize
        nbytes = numel * elem_size
        layout.append((name, tuple(param.shape), dtype, offset, numel))
        offset += nbytes
    return layout


def _flatten_params_into_buffer(
    module: nn.Module,
    buffer: torch.Tensor,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
) -> None:
    """Copy module parameters into a uint8 *buffer* according to *layout*."""
    params = dict(module.named_parameters())
    for name, _shape, dtype, offset_bytes, numel in layout:
        param = params[name]
        elem_size = dtype.itemsize
        nbytes = numel * elem_size
        # Get a typed view into the buffer region.
        region = buffer[offset_bytes : offset_bytes + nbytes].view(dtype)
        region.copy_(param.data.to(dtype).reshape(-1))


def _copy_file_backed_params_into_buffer(
    module: nn.Module | None,
    buffer: torch.Tensor,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
    file_specs: dict[str, FileParamSpec],
    module_param_names: set[str],
    source_path: str,
    file_view: memoryview,
    get_file_view: Any = None,
    get_file_tensor_view: Any = None,
) -> None:
    """Populate slab buffer from a hybrid file/module parameter source.

    For sharded safetensors, *get_file_view* is a callable that returns the
    memoryview for a given file path.  When a ``FileParamSpec`` has a non-empty
    ``file_path``, that path's view is used instead of *file_view*.
    """
    params = dict(module.named_parameters()) if module is not None else {}

    for name, shape, dtype, offset_bytes, numel in layout:
        elem_size = dtype.itemsize
        dst_nbytes = numel * elem_size
        dst_region = buffer[offset_bytes : offset_bytes + dst_nbytes]

        spec = file_specs.get(name)
        if spec is not None:
            # Use per-spec file path for sharded models, else the block-level view
            if spec.file_path and get_file_view is not None:
                resolved_source_path = spec.file_path
                view = get_file_view(spec.file_path)
            else:
                resolved_source_path = source_path
                view = file_view
            if get_file_tensor_view is not None:
                src_typed = get_file_tensor_view(resolved_source_path, spec, view)
            else:
                src_begin = int(spec.file_offset)
                src_end = src_begin + int(spec.source_nbytes)
                src_view = view[src_begin:src_end]
                src_typed = torch.frombuffer(src_view, dtype=spec.source_dtype).reshape(spec.source_shape)
            if spec.source_dtype == dtype:
                dst_region.view(dtype).copy_(src_typed.reshape(-1))
            else:
                # Source dtype differs from runtime block dtype (e.g. F16 file
                # with BF16 runtime). Cast directly into the destination view so
                # we avoid materializing a second large CPU tensor per parameter.
                dst_region.view(dtype).copy_(src_typed.reshape(-1))
            continue

        if name in module_param_names and name in params:
            param = params[name]
            if param.data.numel() == numel:
                dst_region.view(dtype).copy_(param.data.to(dtype).reshape(-1))
                continue
            # Param was evicted (size-0 data) — fall through to zero fill.

        # Fallback for missing/evicted params: zero fill to keep layout deterministic.
        dst_region.zero_()

def _copy_squareq_backed_params_into_buffer(
    module: nn.Module | None,
    buffer: torch.Tensor,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
    squareq_specs: dict[str, SquareQParamSpec],
    module_param_names: set[str],
    squareq_layers: dict[str, Any],
    module_param_cache: dict[str, torch.Tensor] | None = None,
) -> None:
    """Populate slab buffer from a SquareQ BP8 slab + mutable module params.

    Parameters
    ----------
    module_param_cache:
        Optional dict that caches non-SquareQ module param data on CPU.
        On first call, params are stored here. On subsequent calls (after
        eviction empties param data), cached values are used instead.
    """
    params = dict(module.named_parameters()) if module is not None else {}

    for name, shape, dtype, offset_bytes, numel in layout:
        elem_size = dtype.itemsize
        dst_nbytes = numel * elem_size
        dst_region = buffer[offset_bytes : offset_bytes + dst_nbytes]

        spec = squareq_specs.get(name)
        if spec is not None:
            layer = squareq_layers.get(spec.layer_name)
            if isinstance(layer, dict):
                if spec.kind == "weight":
                    qweight = layer.get("qweight")
                    scale = layer.get("scale")
                    zero_point = layer.get("zero_point")
                    if (
                        isinstance(qweight, torch.Tensor)
                        and isinstance(scale, torch.Tensor)
                        and isinstance(zero_point, torch.Tensor)
                    ):
                        if len(shape) == 0:
                            dst_region.zero_()
                            continue
                        out_dim = int(shape[0])
                        in_flat = int(numel // max(out_dim, 1))
                        if out_dim <= 0 or in_flat <= 0:
                            dst_region.zero_()
                            continue

                        q2d = qweight.reshape(int(qweight.shape[0]), -1)
                        q2d = q2d[:out_dim, :in_flat]

                        scale_vec = scale.reshape(-1).to(dtype=torch.float32)[:out_dim]
                        zero_vec = zero_point.reshape(-1).to(dtype=torch.float32)[:out_dim]
                        if scale_vec.numel() != out_dim or zero_vec.numel() != out_dim:
                            dst_region.zero_()
                            continue

                        dequant = (q2d.to(torch.float32) - zero_vec.unsqueeze(1)) * scale_vec.unsqueeze(1)
                        dst_region.copy_(dequant.to(dtype=dtype).reshape(shape).view(torch.uint8).reshape(-1))
                        continue
                elif spec.kind == "bias":
                    bias = layer.get("bias")
                    if isinstance(bias, torch.Tensor):
                        dst_region.view(dtype).copy_(bias.to(dtype=dtype).reshape(-1))
                        continue
                    dst_region.zero_()
                    continue

        if name in module_param_names and name in params:
            param = params[name]
            if param.data.numel() == numel:
                dst_region.view(dtype).copy_(param.data.to(dtype).reshape(-1))
                # Cache on CPU for after eviction
                if module_param_cache is not None and name not in module_param_cache:
                    module_param_cache[name] = param.data.detach().cpu()
                continue
            # Param was evicted (size-0 data) — try cache
            if module_param_cache is not None and name in module_param_cache:
                cached = module_param_cache[name]
                dst_region.view(dtype).copy_(cached.to(dtype).reshape(-1))
                continue

        # Fallback for missing/evicted params: zero fill to keep layout deterministic.
        dst_region.zero_()


def _restore_params_from_tensor(
    module: nn.Module,
    flat_tensor: torch.Tensor,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
) -> None:
    """Replace module parameter ``.data`` with views into *flat_tensor*.

    *flat_tensor* is the flat contiguous buffer (on GPU or host).
    After this call, the module's forward pass uses *flat_tensor*'s memory.
    """
    # Build a lookup for sub-modules by walking named_parameters to get the
    # module/attr pairs.  named_parameters returns "layer.weight" etc.
    # We need to find the parent module and the attribute name.
    for name, shape, dtype, offset_bytes, numel in layout:
        elem_size = dtype.itemsize
        nbytes = numel * elem_size
        view = flat_tensor[offset_bytes : offset_bytes + nbytes].view(dtype).reshape(shape)
        # Navigate to the parameter and replace its .data.
        _set_param_data(module, name, view)


def _restore_file_backed_param_to_cpu_view(
    module: nn.Module,
    dotted_name: str,
    spec: FileParamSpec,
    *,
    source_path: str,
    runtime_dtype: torch.dtype,
    base_file_view: memoryview | None,
    get_file_view: Any = None,
    get_file_tensor_view: Any = None,
) -> bool:
    """Restore a frozen file-backed parameter as a CPU tensor view.

    This keeps the original shape available during backward-phase eviction
    without materializing a private CPU clone of the parameter data.
    """
    if spec.file_path and get_file_view is not None:
        resolved_source_path = spec.file_path
        file_view = get_file_view(spec.file_path)
    else:
        resolved_source_path = source_path
        file_view = base_file_view
    if file_view is None:
        return False

    if get_file_tensor_view is not None:
        src_typed = get_file_tensor_view(resolved_source_path, spec, file_view)
    else:
        src_begin = int(spec.file_offset)
        src_end = src_begin + int(spec.source_nbytes)
        src_view = file_view[src_begin:src_end]
        src_typed = torch.frombuffer(src_view, dtype=spec.source_dtype).reshape(spec.source_shape)
    if spec.source_dtype != runtime_dtype:
        src_typed = src_typed.to(dtype=runtime_dtype)
    _set_param_data(module, dotted_name, src_typed)
    return True


def _get_param_data(module: nn.Module, dotted_name: str) -> torch.Tensor:
    """Return ``module.<dotted_name>.data`` following dot-separated path."""
    parts = dotted_name.split(".")
    current = module
    for part in parts[:-1]:
        current = getattr(current, part)
    return getattr(current, parts[-1]).data


def _set_param_data(module: nn.Module, dotted_name: str, data: torch.Tensor) -> None:
    """Set ``module.<dotted_name>.data = data`` following dot-separated path."""
    parts = dotted_name.split(".")
    current = module
    for part in parts[:-1]:
        current = getattr(current, part)
    param = getattr(current, parts[-1])
    try:
        param.data = data
    except RuntimeError:
        # Tensor type mismatch (e.g. meta-device param receiving CUDA data).
        # Replace the parameter entirely.
        new_param = nn.Parameter(data, requires_grad=param.requires_grad)
        setattr(current, parts[-1], new_param)


def _detach_params(
    module: nn.Module,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
) -> None:
    """Replace each parameter's ``.data`` with a size-0 empty tensor.

    This releases all views into the contiguous GPU buffer so that
    setting ``gpu_tensor = None`` actually frees the GPU storage.
    Without this, parameter views keep the underlying storage alive.
    """
    for name, _shape, dtype, _offset_bytes, _numel in layout:
        _set_param_data(module, name, torch.empty(0, dtype=dtype))


# ── scheduler ────────────────────────────────────────────────────────────


class StagehandScheduler:
    """Orchestrates block prefetching and eviction each training step.

    Parameters
    ----------
    registry:
        Immutable block registry.
    residency:
        Mutable residency map tracking block states.
    transfer_engine:
        Async transfer engine for H2D/D2H copies.
    budget:
        Budget manager for VRAM watermark checks.
    policy:
        Prefetch/eviction policy (e.g. StaticLookaheadPolicy).
    guards:
        Numeric guard for NaN/Inf checks.  May be *None*.
    telemetry:
        Telemetry recorder.
    config:
        Runtime configuration.
    inference_mode:
        If *True*, eviction always uses ``save_back=False`` (frozen blocks,
        no gradients to preserve).
    """

    def __init__(
        self,
        registry: BlockRegistry,
        residency: ResidencyMap,
        transfer_engine: AsyncTransferEngine,
        budget: BudgetLike,
        policy: StaticLookaheadPolicy,
        guards: GuardsLike | None,
        telemetry: StagehandTelemetry,
        config: StagehandConfig,
        inference_mode: bool = False,
    ) -> None:
        self._registry = registry
        self._residency = residency
        self._engine = transfer_engine
        self._budget = budget
        self._policy = policy
        self._guards = guards
        self._telemetry = telemetry
        self._config = config
        self._inference_mode = inference_mode

        self._current_step: int = 0
        self._cursor: int = 0
        self._backward_phase: bool = False
        self._defer_backward_eviction: bool = False

        # Ordered list of block entries for exec_order lookup.
        self._ordered_blocks: list[BlockEntry] = []
        self._total_blocks: int = 0
        self._order_to_id: dict[int, str] = {}
        self.refresh_registry_snapshot()

        # Track pending transfer handles per block.
        self._pending_handles: dict[str, TransferHandle] = {}

        # Conductor bandwidth adapter plumbing (Phase F).
        self._conductor_adapter: StagehandConductorAdapter | None = None
        self._squareq_adapter: SquareQConductorAdapter | None = None
        self._eriquant_adapter: EriQuantAdapter | None = None
        self._pending_bw_tokens: dict[str, tuple[Any, float]] = {}  # block_id -> (token, t0)

        # Opened safetensors mmaps keyed by absolute path.
        self._file_maps: dict[str, tuple[object, mmap.mmap, memoryview]] = {}
        # Cached typed tensor views over mmap-backed safetensors regions.
        self._file_tensor_views: dict[
            tuple[str, int, int, torch.dtype, tuple[int, ...]],
            torch.Tensor,
        ] = {}
        # Loaded SquareQ slab layer dictionaries keyed by absolute path.
        self._squareq_layers: dict[str, dict[str, Any]] = {}
        # Per-block cache for non-SquareQ module params (survives eviction).
        self._module_param_caches: dict[str, dict[str, torch.Tensor]] = {}

    # ── step lifecycle ────────────────────────────────────────────────

    def refresh_registry_snapshot(self) -> None:
        """Refresh cached block-order metadata from the registry."""
        self._ordered_blocks = self._registry.blocks_in_order()
        self._total_blocks = len(self._ordered_blocks)
        self._order_to_id = {
            entry.exec_order: entry.block_id for entry in self._ordered_blocks
        }

    def begin_step(self, step: int) -> None:
        """Initialize state for a new training step."""
        self._current_step = step
        self._cursor = 0
        self._backward_phase = False
        self._telemetry.begin_step(step)

    def end_step(self) -> None:
        """Finalize the current step — reap transfers, update telemetry."""
        self._backward_phase = False
        self._engine.reap()
        self._reconcile_file_backed_grads()
        self._telemetry.end_step()

    def enter_backward_phase(self) -> None:
        """Mark scheduler callbacks as running from backward hooks."""
        self._backward_phase = True

    def exit_backward_phase(self) -> None:
        """Leave backward-hook phase tracking."""
        self._backward_phase = False

    def set_defer_backward_eviction(self, enabled: bool) -> None:
        """Delay eager post-block eviction until the next pre-block pass."""
        self._defer_backward_eviction = enabled

    def _reconcile_file_backed_grads(self) -> None:
        """Ensure CPU-resident mutable params have CPU grads at step boundary."""
        for entry in self._ordered_blocks:
            if not entry.file_backed:
                continue
            module = entry.module_ref()
            if module is None:
                continue
            mutable_names = set(entry.module_param_names)
            if not mutable_names:
                continue
            for name, param in module.named_parameters():
                if name not in mutable_names:
                    continue
                if param.device.type != "cpu":
                    continue
                if param.grad is not None and param.grad.device.type != "cpu":
                    param.grad = param.grad.to("cpu", non_blocking=True)

    def _get_file_view(self, source_path: str) -> memoryview:
        key = str(Path(source_path).expanduser())
        cached = self._file_maps.get(key)
        if cached is not None:
            return cached[2]

        handle = Path(key).open("rb")
        mm = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        view = memoryview(mm)
        self._file_maps[key] = (handle, mm, view)
        return view

    def _get_file_tensor_view(
        self,
        source_path: str,
        spec: FileParamSpec,
        file_view: memoryview | None = None,
    ) -> torch.Tensor:
        """Return a cached typed tensor view over a safetensors parameter region."""
        resolved_path = str(Path(source_path).expanduser())
        key = (
            resolved_path,
            int(spec.file_offset),
            int(spec.source_nbytes),
            spec.source_dtype,
            spec.source_shape,
        )
        cached = self._file_tensor_views.get(key)
        if cached is not None:
            return cached

        view = file_view if file_view is not None else self._get_file_view(resolved_path)
        src_begin = int(spec.file_offset)
        src_end = src_begin + int(spec.source_nbytes)
        src_view = view[src_begin:src_end]
        typed = torch.frombuffer(src_view, dtype=spec.source_dtype).reshape(spec.source_shape)
        self._file_tensor_views[key] = typed
        return typed

    def _get_squareq_layers(self, source_path: str) -> dict[str, Any]:
        key = str(Path(source_path).expanduser())
        cached = self._squareq_layers.get(key)
        if cached is not None:
            return cached

        payload = None
        for kwargs in (
            {"weights_only": True, "mmap": True},
            {"weights_only": True},
        ):
            try:
                payload = torch.load(key, map_location="cpu", **kwargs)
                break
            except TypeError:
                continue
        if payload is None:
            payload = torch.load(key, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid SquareQ slab payload at {key}")
        layers = payload.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"SquareQ slab missing 'layers' map at {key}")

        self._squareq_layers[key] = layers
        return layers

    def _get_squareq_v2_layers(self, source_path: str) -> dict[str, Any]:
        """Load V2 SquareQ layers from safetensors + JSON manifest sidecar.

        Derives the manifest path from the safetensors path
        (``foo.safetensors`` → ``foo.manifest.json``) and delegates to
        ``get_squareq_v2_layers``.  Results are cached in
        ``_squareq_layers`` keyed by absolute path.
        """
        key = str(Path(source_path).expanduser())
        cached = self._squareq_layers.get(key)
        if cached is not None:
            return cached

        manifest_path = str(Path(key).with_suffix(".manifest.json"))
        try:
            from serenity.squareq.stagehand import get_squareq_v2_layers
        except ImportError:
            from squareq.bridge import get_squareq_v2_layers  # standalone fallback

        layers = get_squareq_v2_layers(key, manifest_path)
        self._squareq_layers[key] = layers
        return layers

    def close(self) -> None:
        """Release open file-backed mmap resources."""
        self._file_tensor_views.clear()
        for handle, mm, view in self._file_maps.values():
            try:
                view.release()
            except Exception:
                pass
            try:
                mm.close()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass
        self._file_maps.clear()
        self._squareq_layers.clear()
        self._module_param_caches.clear()

    # ── conductor bandwidth adapter plumbing (Phase F) ──────────────

    def set_conductor_adapter(self, adapter: StagehandConductorAdapter) -> None:
        """Set the Stagehand conductor adapter for bandwidth-aware scheduling."""
        self._conductor_adapter = adapter

    def set_squareq_adapter(self, adapter: SquareQConductorAdapter) -> None:
        """Set the SquareQ conductor adapter for bandwidth-aware scheduling."""
        self._squareq_adapter = adapter

    def set_eriquant_adapter(self, adapter: EriQuantAdapter) -> None:
        """Register an EriQuant StagehandAdapter for quantized block management."""
        self._eriquant_adapter = adapter

    def _acquire_bw_token(
        self, block_id: str, estimated_bytes: int, speculative: bool = False,
    ) -> bool:
        """Acquire a bandwidth token for a block transfer.

        Returns True if the transfer should proceed, False if it should be
        skipped (only possible for speculative prefetches).
        """
        if self._conductor_adapter is None:
            return True

        try:
            # Route to SquareQ adapter for squareq-backed blocks.
            block_entry = self._registry.get(block_id)
            if block_entry.squareq_backed and self._squareq_adapter is not None:
                if speculative:
                    result = self._squareq_adapter.request_lookahead(
                        estimated_bytes=estimated_bytes,
                        block_id=block_id,
                        step_id=self._current_step,
                    )
                else:
                    result = self._squareq_adapter.request_slab_load(
                        estimated_bytes=estimated_bytes,
                        block_id=block_id,
                        step_id=self._current_step,
                    )
            else:
                if speculative:
                    result = self._conductor_adapter.request_prefetch(
                        block_id=block_id,
                        estimated_bytes=estimated_bytes,
                        step_id=self._current_step,
                    )
                else:
                    result = self._conductor_adapter.request_block_load(
                        block_id=block_id,
                        estimated_bytes=estimated_bytes,
                        step_id=self._current_step,
                    )

            if result is None:
                # Bandwidth scheduling disabled in adapter — proceed.
                return True

            from serenity.memory.conductor.bandwidth import AcquireStatus  # type: ignore[import-not-found]
            if result.status == AcquireStatus.ACQUIRED and result.token is not None:
                self._pending_bw_tokens[block_id] = (result.token, time.monotonic())
                return True

            # Non-ACQUIRED: skip speculative, proceed for required.
            if speculative:
                return False
            log.debug(
                "Bandwidth non-ACQUIRED for required block %s (status=%s), proceeding",
                block_id, result.status.value,
            )
            return True

        except Exception:
            log.warning(
                "Bandwidth token acquire failed for block %s, proceeding",
                block_id, exc_info=True,
            )
            return True

    def _complete_bw_token(self, block_id: str, actual_bytes: int) -> None:
        """Complete a bandwidth token after a transfer finishes."""
        pending = self._pending_bw_tokens.pop(block_id, None)
        if pending is None:
            return

        token, t0 = pending
        duration_ms = (time.monotonic() - t0) * 1000.0

        try:
            conductor = self._conductor_adapter._conductor
            if conductor is not None and conductor.transfer is not None:
                conductor.transfer.complete_transfer(token, actual_bytes, duration_ms)
                conductor.transfer.release_token(token)
        except Exception:
            log.warning(
                "Bandwidth token completion failed for block %s",
                block_id, exc_info=True,
            )

    # ── per-block hooks ───────────────────────────────────────────────

    def before_block(self, block_id: str) -> None:
        """Called before a block computes.  Ensures GPU residency."""
        state = self._residency.get_state(block_id)

        if state == BlockState.GPU_READY:
            self._telemetry.record_prefetch_hit()
        elif state == BlockState.PREFETCHING:
            # Wait for in-progress transfer — this is a stall.
            t0 = time.monotonic()
            handle = self._pending_handles.get(block_id)
            if handle is not None:
                self._engine.wait(handle)
                self._pending_handles.pop(block_id, None)
            self._finalize_gpu_load(block_id)
            stall_ms = (time.monotonic() - t0) * 1000.0
            self._telemetry.record_stall(stall_ms)
            self._telemetry.record_prefetch_miss()
            log.debug("Stall on block %s: %.2f ms", block_id, stall_ms)
        else:
            # Block not on GPU at all — hard stall. Load it now.
            t0 = time.monotonic()
            self._load_block_to_gpu(block_id)
            # If it became PREFETCHING, wait for it.
            if self._residency.get_state(block_id) == BlockState.PREFETCHING:
                handle = self._pending_handles.get(block_id)
                if handle is not None:
                    self._engine.wait(handle)
                    self._pending_handles.pop(block_id, None)
                self._finalize_gpu_load(block_id)
            stall_ms = (time.monotonic() - t0) * 1000.0
            self._telemetry.record_stall(stall_ms)
            self._telemetry.record_prefetch_miss()
            if _STALL_LOGS_ENABLED:
                log.warning("Hard stall on block %s (was %s): %.2f ms", block_id, state.value, stall_ms)

        # Increment refcount — block is now in use.
        self._residency.increment_ref(block_id)
        entry = self._residency.get_entry(block_id)
        entry.last_used_step = self._current_step

        # Advance cursor.
        self._cursor += 1

        # Eviction pass BEFORE prefetching — free VRAM for upcoming blocks.
        if self._budget.above_high_watermark():
            self._run_eviction(ignore_cooldown=True)

        # Issue prefetches for lookahead window.
        self._prefetch_ahead()

        # Second eviction pass after prefetching in case we're still over budget.
        if self._budget.above_high_watermark():
            self._run_eviction(ignore_cooldown=True)

    def after_block(self, block_id: str, output: torch.Tensor | None = None) -> None:
        """Called after a block computes.  Releases refcount and runs eviction."""
        self._residency.decrement_ref(block_id)
        entry = self._residency.get_entry(block_id)
        entry.last_used_step = self._current_step

        # Numeric guard check.
        if self._guards is not None and output is not None and self._config.nan_inf_check:
            nan_count, inf_count = self._guards.check_output(
                output, block_id, self._current_step,
            )
            if nan_count > 0 or inf_count > 0:
                self._telemetry.record_nan_inf(nan_count, inf_count)

        # Evict old blocks eagerly after each block completes.  This keeps
        # VRAM usage bounded to roughly (prefetch_window + 1) blocks instead
        # of accumulating all blocks on GPU until the watermark is hit.
        # ignore_cooldown=True: within a step, completed blocks (refcount 0)
        # should be evictable immediately — cooldown is for cross-step thrashing.
        if (
            self._budget.above_high_watermark()
            and not (self._backward_phase and self._defer_backward_eviction)
        ):
            self._run_eviction(ignore_cooldown=True)

    # ── prefetch logic ────────────────────────────────────────────────

    def _prefetch_ahead(self) -> None:
        """Issue H2D transfers for blocks in the lookahead window."""
        # Build residency state map for the policy.
        residency_states: dict[str, str] = {}
        for entry in self._ordered_blocks:
            residency_states[entry.block_id] = self._residency.get_state(
                entry.block_id
            ).value

        indices = self._policy.blocks_to_prefetch(
            cursor=self._cursor - 1,  # cursor was already incremented
            total_blocks=self._total_blocks,
            residency_states=residency_states,
        )

        for idx in indices:
            if idx >= self._total_blocks:
                break
            block_id = self._ordered_blocks[idx].block_id
            state = self._residency.get_state(block_id)

            if state in (BlockState.GPU_READY, BlockState.PREFETCHING):
                continue

            # Don't prefetch if VRAM is already near the high watermark —
            # block will be demand-loaded by before_block when actually needed.
            if not self._budget.can_prefetch():
                break

            self._load_block_to_gpu(block_id, speculative=True)

    def _load_block_to_gpu(self, block_id: str, speculative: bool = False) -> None:
        """Ensure block progresses toward GPU_READY."""
        state = self._residency.get_state(block_id)

        if state == BlockState.GPU_READY:
            return
        if state == BlockState.PREFETCHING:
            return

        block_entry = self._registry.get(block_id)

        # Bandwidth gate: acquire token before any transfer.
        if not self._acquire_bw_token(block_id, block_entry.size_bytes, speculative):
            return  # Speculative prefetch denied — skip.
        res_entry = self._residency.get_entry(block_id)

        # EriQuant-backed blocks: delegate to the EriQuant adapter which
        # manages frozen quantized weights.  The adapter's on_prefetch()
        # calls module.to(device) on the frozen (smaller) representation.
        if block_entry.eriquant_backed:
            if self._eriquant_adapter is None:
                log.error("Block %r is eriquant-backed but no adapter set — skipping", block_id)
                return
            if state == BlockState.UNLOADED:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._eriquant_adapter.on_prefetch(block_id, device)
                # Walk through valid state transitions for residency tracking.
                self._residency.transition(block_id, BlockState.HOST_STAGED)
                self._residency.transition(block_id, BlockState.PREFETCHING)
                self._residency.transition(block_id, BlockState.GPU_READY)
                self._telemetry.record_h2d(block_entry.size_bytes)
                self._complete_bw_token(block_id, block_entry.size_bytes)
            return

        # Module-backed blocks (not file/squareq): params live on the module
        # itself (on CPU after eviction).  Move directly to GPU without going
        # through the pinned-slab pipeline, which can't handle blocks larger
        # than a single slab for module-backed data.
        is_module_backed = not block_entry.file_backed and not block_entry.squareq_backed and not block_entry.eriquant_backed
        if is_module_backed and state == BlockState.UNLOADED:
            self._load_module_backed_direct(block_id, block_entry, res_entry)
            return

        if state == BlockState.UNLOADED:
            # Stage to host first, then submit H2D.
            slab = self._stage_block_to_host(block_entry, res_entry)
            res_entry.host_slab = slab
            state = self._residency.get_state(block_id)  # now HOST_STAGED

        if state == BlockState.HOST_STAGED:
            # Allocate GPU tensor if needed.
            if res_entry.gpu_tensor is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                numel = block_entry.size_bytes // block_entry.dtype.itemsize
                res_entry.gpu_tensor = torch.empty(
                    numel, dtype=block_entry.dtype, device=device
                )

            # Submit H2D transfer.
            handle = self._engine.submit_h2d(
                block_id=block_id,
                host_slab=res_entry.host_slab,
                gpu_dest=res_entry.gpu_tensor,
            )
            self._pending_handles[block_id] = handle
            self._residency.transition(block_id, BlockState.PREFETCHING)
            self._telemetry.record_h2d(block_entry.size_bytes)

    def _stage_block_to_host(
        self, block_entry: BlockEntry, res_entry: ResidencyEntry,
    ) -> PinnedSlab:
        """Acquire a slab and copy block parameters into it.

        Builds a contiguous param layout, flattens all module parameters
        into the slab buffer, and stores the layout on the residency entry
        so GPU-side restoration can reconstruct individual parameter views.
        """
        slab = self._engine._pool.acquire(block_entry.size_bytes)

        # Resolve the module from the weak reference.
        module = block_entry.module_ref()
        if module is not None:
            if block_entry.squareq_backed:
                layout = list(block_entry.param_layout)
                res_entry.param_layout = layout
                if not isinstance(slab, list):
                    squareq_specs = {spec.param_name: spec for spec in block_entry.squareq_param_specs}
                    module_param_names = set(block_entry.module_param_names)
                    if block_entry.source_format == "squareq_v2":
                        squareq_layers = self._get_squareq_v2_layers(str(block_entry.source_path))
                    else:
                        squareq_layers = self._get_squareq_layers(str(block_entry.source_path))
                    # Per-block cache for non-SquareQ module params that survive eviction
                    block_cache = self._module_param_caches.setdefault(block_entry.block_id, {})
                    _copy_squareq_backed_params_into_buffer(
                        module=module,
                        buffer=slab.buffer,
                        layout=layout,
                        squareq_specs=squareq_specs,
                        module_param_names=module_param_names,
                        squareq_layers=squareq_layers,
                        module_param_cache=block_cache,
                    )
            elif block_entry.file_backed:
                layout = list(block_entry.param_layout)
                res_entry.param_layout = layout
                if not isinstance(slab, list):
                    file_specs = {spec.param_name: spec for spec in block_entry.file_param_specs}
                    module_param_names = set(block_entry.module_param_names)
                    # For sharded safetensors, use per-spec file paths via get_file_view.
                    # For single-file, use the block-level source_path.
                    is_sharded = block_entry.source_format == "safetensors_sharded"
                    if is_sharded:
                        # Sharded: pass a dummy file_view (won't be used) and the resolver
                        file_view = memoryview(b"")
                    else:
                        file_view = self._get_file_view(str(block_entry.source_path))
                    _copy_file_backed_params_into_buffer(
                        module=module,
                        buffer=slab.buffer,
                        layout=layout,
                        file_specs=file_specs,
                        module_param_names=module_param_names,
                        source_path=str(block_entry.source_path),
                        file_view=file_view,
                        get_file_view=self._get_file_view if is_sharded else None,
                        get_file_tensor_view=self._get_file_tensor_view,
                    )
            elif any(True for _ in module.parameters()):
                layout = _build_param_layout(module, block_entry.dtype)
                res_entry.param_layout = layout
                if not isinstance(slab, list):
                    _flatten_params_into_buffer(module, slab.buffer, layout)
            else:
                # Module has no parameters.
                res_entry.param_layout = None
                if not isinstance(slab, list):
                    slab.buffer[:] = 0
        else:
            # Module has been garbage-collected or has no parameters.
            # Zero-fill as a fallback.
            res_entry.param_layout = None
            if not isinstance(slab, list):
                slab.buffer[:] = 0

        self._residency.transition(block_entry.block_id, BlockState.HOST_STAGED)
        return slab

    def _load_module_backed_direct(
        self,
        block_id: str,
        block_entry: BlockEntry,
        res_entry: ResidencyEntry,
    ) -> None:
        """Load module-backed params to GPU without pinned slabs.

        Module-backed blocks store their data as CPU tensors on the module
        itself (set during eviction).  We build a param layout, allocate a
        contiguous GPU tensor (so the VRAM budget tracks it), flatten CPU
        params into it, then restore param views — same as the normal path
        but skipping the pinned-slab intermediary that can't handle multi-slab
        blocks for module-backed data.
        """
        module = block_entry.module_ref()
        if module is None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build param layout from the current CPU params.
        layout = _build_param_layout(module, block_entry.dtype)
        res_entry.param_layout = layout

        # Allocate contiguous GPU buffer (tracked by VRAM budget).
        numel = block_entry.size_bytes // block_entry.dtype.itemsize
        gpu_tensor = torch.empty(numel, dtype=block_entry.dtype, device=device)
        res_entry.gpu_tensor = gpu_tensor

        # Flatten CPU params directly into the GPU tensor.
        gpu_bytes = gpu_tensor.view(torch.uint8)
        params = dict(module.named_parameters())
        for name, _shape, dtype, offset_bytes, param_numel in layout:
            param = params[name]
            elem_size = dtype.itemsize
            nbytes = param_numel * elem_size
            region = gpu_bytes[offset_bytes : offset_bytes + nbytes].view(dtype)
            region.copy_(param.data.to(dtype).reshape(-1))

        # Restore module params as views into the contiguous GPU tensor.
        _restore_params_from_tensor(module, gpu_bytes, layout)

        # Move accumulated .grad tensors to GPU.
        for param in module.parameters():
            if param.grad is not None and param.grad.device.type != device:
                param.grad = param.grad.to(device, non_blocking=True)

        # Walk through valid state transitions:
        # UNLOADED → HOST_STAGED → PREFETCHING → GPU_READY
        self._residency.transition(block_id, BlockState.HOST_STAGED)
        self._residency.transition(block_id, BlockState.PREFETCHING)
        self._residency.transition(block_id, BlockState.GPU_READY)
        self._telemetry.record_h2d(block_entry.size_bytes)
        self._complete_bw_token(block_id, block_entry.size_bytes)


    def _finalize_gpu_load(self, block_id: str) -> None:
        """After H2D transfer completes, restore module params from GPU tensor.

        Transitions the block to GPU_READY, uses the stored param_layout
        to replace each module parameter's ``.data`` with a view into the
        contiguous GPU buffer, then releases the host slab back to the pool
        (data now lives on GPU; a fresh slab will be acquired if save-back
        eviction is needed later).
        """
        self._residency.transition(block_id, BlockState.GPU_READY)
        block_entry = self._registry.get(block_id)
        self._complete_bw_token(block_id, block_entry.size_bytes)
        res_entry = self._residency.get_entry(block_id)

        if res_entry.param_layout is not None and res_entry.gpu_tensor is not None:
            module = block_entry.module_ref()
            if module is not None:
                # Reinterpret the flat GPU tensor as uint8 so we can index by
                # byte offset (matching _flatten_params_into_buffer layout).
                gpu_bytes = res_entry.gpu_tensor.view(torch.uint8)
                _restore_params_from_tensor(module, gpu_bytes, res_entry.param_layout)

                # Move accumulated .grad tensors to GPU so gradient accumulation
                # (step N+1 backward adding to step N grads) doesn't hit a
                # device mismatch between GPU activations and CPU .grad.
                gpu_device = res_entry.gpu_tensor.device
                for param in module.parameters():
                    if param.grad is not None and param.grad.device != gpu_device:
                        param.grad = param.grad.to(gpu_device, non_blocking=True)

        # Release the host slab — data is now on GPU.  A fresh slab will be
        # acquired if we later need to evict with save-back (D2H).
        if res_entry.host_slab is not None:
            self._engine._pool.release(res_entry.host_slab)
            res_entry.host_slab = None

    # ── eviction logic ────────────────────────────────────────────────

    def _run_eviction(self, ignore_cooldown: bool = False) -> None:
        """Evict blocks until VRAM is below the low watermark.

        Spec rules enforced:
        - Never evict a block with refcount > 0 (handled by eviction_candidates).
        - Never evict a block within the prefetch window.
        - Evict in descending score order until below vram_low_watermark.
        - Eviction cooldown respected unless *ignore_cooldown* is True.

        Within a single training step, blocks that completed their forward
        pass have refcount 0 and should be evictable immediately.  The
        cooldown (designed to prevent cross-step thrashing) would block
        this because ``last_used_step == current_step``.  Callers from
        ``after_block`` pass ``ignore_cooldown=True`` to allow immediate
        eviction of completed blocks.
        """
        cooldown = 0 if ignore_cooldown else self._policy.eviction_cooldown_steps
        candidates = self._residency.eviction_candidates(
            current_step=self._current_step,
            cooldown_steps=cooldown,
        )

        # Build the set of block_ids within the prefetch window -- these are
        # protected from eviction per spec Section 2.4.2.
        prefetch_window_ids: set[str] = set()
        cursor_for_policy = max(self._cursor - 1, 0)
        for offset in range(1, self._policy.prefetch_window + 1):
            idx = cursor_for_policy + offset
            if idx < self._total_blocks:
                prefetch_window_ids.add(self._ordered_blocks[idx].block_id)

        # Score and sort — highest score = evict first.
        scored: list[tuple[float, str]] = []
        for bid, entry in candidates:
            if bid in prefetch_window_ids:
                continue  # Never evict blocks within the prefetch window.
            block_entry = self._registry.get(bid)
            score = self._policy.score_for_eviction(
                block_id=bid,
                current_cursor=self._cursor,
                entry_exec_order=block_entry.exec_order,
                total_blocks=self._total_blocks,
                size_bytes=block_entry.size_bytes,
            )
            scored.append((score, bid))

        scored.sort(reverse=True)

        for _score, bid in scored:
            if self._budget.below_low_watermark():
                break
            # In inference mode, never save back (blocks are frozen).
            save_back = not self._inference_mode
            self._evict_block(bid, save_back=save_back)
            self._telemetry.record_eviction()

    def _evict_block(self, block_id: str, save_back: bool = False) -> None:
        """Evict a block from GPU.

        Parameters
        ----------
        save_back:
            If *True*, D2H the GPU tensor before freeing (for blocks with
            gradients that may have been updated by the optimizer).
            If *False*, just free the GPU tensor.
        """
        block_entry = self._registry.get(block_id)
        res_entry = self._residency.get_entry(block_id)
        module = block_entry.module_ref()

        # EriQuant-backed blocks: delegate to the adapter.
        if block_entry.eriquant_backed:
            if self._eriquant_adapter is None:
                log.error("Block %r is eriquant-backed but no adapter set — skipping evict", block_id)
                return
            self._residency.transition(block_id, BlockState.EVICTING)
            self._eriquant_adapter.on_offload(block_id, "cpu")
            res_entry.gpu_tensor = None
            if res_entry.host_slab is not None:
                self._engine._pool.release(res_entry.host_slab)
                res_entry.host_slab = None
            res_entry.param_layout = None
            self._residency.transition(block_id, BlockState.UNLOADED)
            return

        if save_back:
            # Module-backed blocks: params live on the module itself.  Copy
            # each GPU param to a standalone CPU tensor — no slab needed.
            is_module_backed = not block_entry.file_backed and not block_entry.squareq_backed and not block_entry.eriquant_backed
            if is_module_backed:
                self._residency.transition(block_id, BlockState.EVICTING)
                if module is not None:
                    for param in module.parameters():
                        if param.data.device.type != "cpu":
                            param.data = param.data.to("cpu", non_blocking=True).clone()
                        if param.grad is not None and param.grad.device.type != "cpu":
                            param.grad = param.grad.to("cpu", non_blocking=True)
                res_entry.gpu_tensor = None
                if res_entry.host_slab is not None:
                    self._engine._pool.release(res_entry.host_slab)
                    res_entry.host_slab = None
                res_entry.param_layout = None
                self._residency.transition(block_id, BlockState.UNLOADED)
                return

            # File-backed blocks keep frozen base params on disk. On eviction,
            # only mutable params (e.g. LoRA) need CPU save-back.
            if block_entry.file_backed:
                self._residency.transition(block_id, BlockState.EVICTING)

                mutable_names = set(block_entry.module_param_names)
                keep_mutable_on_gpu = self._defer_backward_eviction and not self._backward_phase
                file_specs = {spec.param_name: spec for spec in block_entry.file_param_specs}
                base_file_view: memoryview | None = None
                if any(not spec.file_path for spec in block_entry.file_param_specs):
                    base_file_view = self._get_file_view(str(block_entry.source_path))
                if module is not None and res_entry.param_layout is not None:
                    for name, _shape, dtype, _offset, _numel in res_entry.param_layout:
                        if name in mutable_names:
                            live_data = _get_param_data(module, name)
                            target_device = live_data.device if keep_mutable_on_gpu else torch.device("cpu")
                            copied = live_data.to(target_device, non_blocking=True).clone()
                            _set_param_data(module, name, copied)
                        else:
                            restored = False
                            if self._backward_phase and self._defer_backward_eviction:
                                spec = file_specs.get(name)
                                if spec is not None:
                                    restored = _restore_file_backed_param_to_cpu_view(
                                        module,
                                        name,
                                        spec,
                                        source_path=str(block_entry.source_path),
                                        runtime_dtype=dtype,
                                        base_file_view=base_file_view,
                                        get_file_view=self._get_file_view,
                                        get_file_tensor_view=self._get_file_tensor_view,
                                    )
                            if not restored:
                                _set_param_data(module, name, torch.empty(0, dtype=dtype))

                    for name, param in module.named_parameters():
                        if name in mutable_names and param.grad is not None and param.grad.device.type != "cpu":
                            param.grad = param.grad.to("cpu", non_blocking=True)

                res_entry.gpu_tensor = None
                if res_entry.host_slab is not None:
                    self._engine._pool.release(res_entry.host_slab)
                    res_entry.host_slab = None
                res_entry.param_layout = None
                self._residency.transition(block_id, BlockState.UNLOADED)
                return

            # D2H: GPU_READY -> EVICTING -> HOST_STAGED -> (release slab) -> UNLOADED.
            # We acquire a temporary slab for the DMA transfer, copy back into
            # regular CPU tensors, then release the slab immediately.  This
            # prevents slab exhaustion when many blocks are evicted in a single
            # forward pass (40 blocks but only 8 slabs).
            self._residency.transition(block_id, BlockState.EVICTING)
            if res_entry.gpu_tensor is not None:
                slab = self._engine._pool.acquire(block_entry.size_bytes)
                handle = self._engine.submit_d2h(
                    block_id=block_id,
                    gpu_src=res_entry.gpu_tensor,
                    host_slab=slab,
                )
                self._engine.wait(handle)
                self._telemetry.record_d2h(
                    res_entry.gpu_tensor.numel() * res_entry.gpu_tensor.element_size()
                )

                # Restore module parameters from the slab into CPU views.
                if (
                    module is not None
                    and res_entry.param_layout is not None
                ):
                    _restore_params_from_tensor(
                        module, slab.buffer, res_entry.param_layout,
                    )
                    # Detach params from the slab: copy each param.data to a
                    # standalone CPU tensor so the slab can be freed.
                    for name, _shape, _dtype, _offset, _numel in res_entry.param_layout:
                        _set_param_data(
                            module, name,
                            _get_param_data(module, name).clone(),
                        )

                # Release the slab back to the pool immediately.
                self._engine._pool.release(slab)
                res_entry.host_slab = None

            # Move any .grad tensors from GPU to CPU.  During backward,
            # autograd creates .grad tensors on the same device as the
            # parameter (GPU while loaded).  The optimizer needs them on
            # the same device as param.data (CPU after eviction).
            if module is not None:
                for param in module.parameters():
                    if param.grad is not None and param.grad.device.type != "cpu":
                        param.grad = param.grad.to("cpu", non_blocking=True)

            # Free GPU tensor — data is now safely in CPU params.
            res_entry.gpu_tensor = None
            # Transition to UNLOADED (not HOST_STAGED) since no slab is held.
            # _load_block_to_gpu will re-stage from the CPU params next time.
            self._residency.transition(block_id, BlockState.UNLOADED)
        else:
            # No save-back: GPU_READY -> GPU_FREEING -> UNLOADED.
            self._residency.transition(block_id, BlockState.GPU_FREEING)

            # For module-backed blocks (no file source), the module's own
            # parameters are the ONLY source of data.  _detach_params would
            # replace param.data with empty(0) tensors, destroying shapes AND
            # data.  On the next _stage_block_to_host, _build_param_layout
            # would see numel=0 for every param and produce a broken layout.
            #
            # Fix: copy GPU params back to standalone CPU tensors before
            # dropping the GPU buffer.  This preserves both shape and data
            # for re-staging while still freeing GPU memory.
            is_module_backed = not block_entry.file_backed and not block_entry.squareq_backed and not block_entry.eriquant_backed
            if module is not None and res_entry.param_layout is not None:
                if is_module_backed:
                    # Save params to CPU so re-staging can read them back.
                    for name, _shape, _dtype, _offset, _numel in res_entry.param_layout:
                        gpu_data = _get_param_data(module, name)
                        _set_param_data(module, name, gpu_data.to("cpu", non_blocking=True).clone())
                elif block_entry.file_backed:
                    resident_names = set(block_entry.module_param_names)
                    keep_resident_on_gpu = self._defer_backward_eviction and not self._backward_phase
                    for name, _shape, dtype, _offset, _numel in res_entry.param_layout:
                        if name in resident_names:
                            gpu_data = _get_param_data(module, name)
                            target_device = gpu_data.device if keep_resident_on_gpu else torch.device("cpu")
                            _set_param_data(module, name, gpu_data.to(target_device, non_blocking=True).clone())
                        else:
                            _set_param_data(module, name, torch.empty(0, dtype=dtype))
                else:
                    # SquareQ-backed: data can be re-read from source.
                    _detach_params(module, res_entry.param_layout)

            # Free GPU tensor — now safe because views have been replaced.
            res_entry.gpu_tensor = None
            # Release host slab if held.
            if res_entry.host_slab is not None:
                self._engine._pool.release(res_entry.host_slab)
                res_entry.host_slab = None
            # Clear param layout since slab is gone.
            res_entry.param_layout = None
            self._residency.transition(block_id, BlockState.UNLOADED)
