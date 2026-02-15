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
    file_view: memoryview,
    get_file_view: Any = None,
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
                view = get_file_view(spec.file_path)
            else:
                view = file_view
            src_begin = int(spec.file_offset)
            src_end = src_begin + int(spec.source_nbytes)
            src_view = view[src_begin:src_end]
            if spec.source_dtype == dtype:
                src_bytes = torch.frombuffer(src_view, dtype=torch.uint8)
                dst_region.copy_(src_bytes)
            else:
                # Source dtype differs from runtime block dtype (e.g. F16 file
                # with BF16 runtime). Decode + cast per parameter.
                src_typed = torch.frombuffer(src_view, dtype=spec.source_dtype).reshape(spec.source_shape)
                cast = src_typed.to(dtype=dtype)
                dst_region.copy_(cast.view(torch.uint8).reshape(-1))
            continue

        if name in module_param_names and name in params:
            param = params[name]
            dst_region.view(dtype).copy_(param.data.to(dtype).reshape(-1))
            continue

        # Fallback for missing params: zero fill to keep layout deterministic.
        dst_region.zero_()


def _copy_squareq_backed_params_into_buffer(
    module: nn.Module | None,
    buffer: torch.Tensor,
    layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]],
    squareq_specs: dict[str, SquareQParamSpec],
    module_param_names: set[str],
    squareq_layers: dict[str, Any],
) -> None:
    """Populate slab buffer from a SquareQ BP8 slab + mutable module params."""
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
                    if zero_point is None:
                        zero_point = layer.get("zero")
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
            dst_region.view(dtype).copy_(param.data.to(dtype).reshape(-1))
            continue

        # Fallback for missing params: zero fill to keep layout deterministic.
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
    param.data = data


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

        # Ordered list of block entries for exec_order lookup.
        self._ordered_blocks: list[BlockEntry] = []
        self._total_blocks: int = 0
        self._order_to_id: dict[int, str] = {}
        self.refresh_registry_snapshot()

        # Track pending transfer handles per block.
        self._pending_handles: dict[str, TransferHandle] = {}
        # Opened safetensors mmaps keyed by absolute path.
        self._file_maps: dict[str, tuple[object, mmap.mmap, memoryview]] = {}
        # Loaded SquareQ slab layer dictionaries keyed by absolute path.
        self._squareq_layers: dict[str, dict[str, Any]] = {}

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
        self._telemetry.begin_step(step)

    def end_step(self) -> None:
        """Finalize the current step — reap transfers, update telemetry."""
        self._engine.reap()
        self._reconcile_file_backed_grads()
        self._telemetry.end_step()

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

    def close(self) -> None:
        """Release open file-backed mmap resources."""
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
        if self._budget.above_high_watermark():
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

            self._load_block_to_gpu(block_id)

    def _load_block_to_gpu(self, block_id: str) -> None:
        """Ensure block progresses toward GPU_READY."""
        state = self._residency.get_state(block_id)

        if state == BlockState.GPU_READY:
            return
        if state == BlockState.PREFETCHING:
            return

        block_entry = self._registry.get(block_id)
        res_entry = self._residency.get_entry(block_id)

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
                    squareq_layers = self._get_squareq_layers(str(block_entry.source_path))
                    _copy_squareq_backed_params_into_buffer(
                        module=module,
                        buffer=slab.buffer,
                        layout=layout,
                        squareq_specs=squareq_specs,
                        module_param_names=module_param_names,
                        squareq_layers=squareq_layers,
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
                        file_view=file_view,
                        get_file_view=self._get_file_view if is_sharded else None,
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

        if save_back:
            # File-backed blocks keep frozen base params on disk. On eviction,
            # only mutable params (e.g. LoRA) need CPU save-back.
            if block_entry.file_backed:
                self._residency.transition(block_id, BlockState.EVICTING)

                mutable_names = set(block_entry.module_param_names)
                if module is not None and res_entry.param_layout is not None:
                    for name, _shape, dtype, _offset, _numel in res_entry.param_layout:
                        if name in mutable_names:
                            cpu_copy = _get_param_data(module, name).to("cpu", non_blocking=True).clone()
                            _set_param_data(module, name, cpu_copy)
                        else:
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
            # Detach module parameters from GPU storage BEFORE dropping
            # gpu_tensor.  Without this, parameter views keep the underlying
            # GPU storage alive and VRAM is never actually freed.
            if module is not None and res_entry.param_layout is not None:
                _detach_params(module, res_entry.param_layout)
            # Free GPU tensor — now safe because no views reference it.
            res_entry.gpu_tensor = None
            # Release host slab if held.
            if res_entry.host_slab is not None:
                self._engine._pool.release(res_entry.host_slab)
                res_entry.host_slab = None
            # Clear param layout since slab is gone.
            res_entry.param_layout = None
            self._residency.transition(block_id, BlockState.UNLOADED)
