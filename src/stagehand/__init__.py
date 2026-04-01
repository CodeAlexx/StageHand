"""Stagehand — bounded block-swapping runtime for Serenity.

Top-level package that re-exports all public types from every module
and provides the :class:`StagehandRuntime` integration API.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import torch
from torch import nn

from stagehand.budget import BudgetManager
from stagehand.config import StagehandConfig
from stagehand.errors import (
    DtypeMismatchError,
    InvalidStateTransitionError,
    StagehandError,
    StagehandOOMError,
    TransferError,
)
from stagehand.guards import NumericGuard
from stagehand.layer import LayerRuntime
from stagehand.pool import PinnedPool, PinnedSlab
from stagehand.registry import BlockEntry, BlockRegistry, SquareQParamSpec
from stagehand.residency import BlockState, ResidencyEntry, ResidencyMap
from stagehand.scheduler import StaticLookaheadPolicy, StagehandScheduler
from stagehand.telemetry import StagehandTelemetry, StepMetrics
from stagehand.transfer import AsyncTransferEngine, TransferHandle
try:
    from stagehand.vmm_backend import VmmManager, VmmModelHandle, is_available as vmm_is_available
except Exception:
    # vmm_backend.py itself should always import fine (only depends on torch),
    # but guard anyway so a broken install never breaks the whole package.
    VmmManager = None  # type: ignore[assignment,misc]
    VmmModelHandle = None  # type: ignore[assignment,misc]
    def vmm_is_available() -> bool: return False  # noqa: E704

__all__ = [
    # config
    "StagehandConfig",
    # errors
    "DtypeMismatchError",
    "InvalidStateTransitionError",
    "StagehandError",
    "StagehandOOMError",
    "TransferError",
    # pool
    "PinnedPool",
    "PinnedSlab",
    # registry
    "BlockEntry",
    "BlockRegistry",
    "SquareQParamSpec",
    # residency
    "BlockState",
    "ResidencyEntry",
    "ResidencyMap",
    # transfer
    "AsyncTransferEngine",
    "TransferHandle",
    # scheduler
    "StaticLookaheadPolicy",
    "StagehandScheduler",
    # budget
    "BudgetManager",
    # guards
    "NumericGuard",
    # telemetry
    "StagehandTelemetry",
    "StepMetrics",
    # runtime
    "StagehandRuntime",
    # layer mode
    "LayerRuntime",
    "layer",
    "wrap",
    "Runtime",
]

log = logging.getLogger(__name__)
_DEBUG_STAGEHAND_SHUTDOWN = str(os.getenv("SERENITY_DEBUG_STAGEHAND_SHUTDOWN", "0")).strip().lower() in {"1", "true", "yes", "on"}


class StagehandRuntime:
    """Top-level Stagehand API.  Integrates with training/inference loops via hooks.

    Parameters
    ----------
    model:
        The PyTorch model containing swappable blocks.
    config:
        Runtime configuration (pool sizes, watermarks, etc.).
    block_pattern:
        Regex pattern matched against module names to identify swappable blocks.
    group:
        Logical group label for registered blocks (e.g. ``"transformer"``).
    dtype:
        Target dtype for parameter storage and transfer.
    inference_mode:
        If *True*, backward hooks are no-ops and eviction never saves back
        to host (blocks are frozen, no gradient state to preserve).
    """

    def __init__(
        self,
        model: nn.Module,
        config: StagehandConfig,
        block_pattern: str = r".*\.block\.\d+$",
        group: str = "transformer",
        dtype: torch.dtype = torch.bfloat16,
        inference_mode: bool = False,
        pool: PinnedPool | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._inference_mode = inference_mode

        # 1. Build registry from model.
        self._registry = BlockRegistry()
        self._registry.build_from_model(
            model, block_pattern=block_pattern, group=group, dtype=dtype,
        )

        # 2. Create pool (or reuse existing) and validate registry against pool capacity.
        if pool is not None:
            self._pool = pool
        else:
            self._pool = PinnedPool(
                total_mb=config.pinned_pool_mb,
                slab_mb=config.pinned_slab_mb,
                alignment=config.pinned_slab_alignment,
            )
        self._registry.validate(pool_capacity_bytes=self._pool.num_slabs * self._pool.slab_bytes)

        # 3. Create residency map, budget manager, guards, telemetry.
        self._residency = ResidencyMap(self._registry)
        self._budget = BudgetManager(
            high_watermark_mb=config.vram_high_watermark_mb,
            low_watermark_mb=config.vram_low_watermark_mb,
        )
        self._guards: NumericGuard | None = None
        if config.nan_inf_check or config.strict_bf16:
            self._guards = NumericGuard(
                strict_bf16=config.strict_bf16,
                fail_on_dtype_promotion=config.fail_on_dtype_promotion,
                nan_inf_check=config.nan_inf_check,
            )
        self._telemetry = StagehandTelemetry(
            enabled=config.telemetry_enabled,
            interval_steps=config.telemetry_interval_steps,
            output_file=config.telemetry_file if config.telemetry_enabled else None,
        )

        # 4. Create transfer engine with a dedicated CUDA stream.
        copy_stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available():
            copy_stream = torch.cuda.Stream(priority=config.copy_stream_priority)
        self._engine = AsyncTransferEngine(
            pool=self._pool,
            max_inflight=config.max_inflight_transfers,
            copy_stream=copy_stream,
        )

        # 5. Create scheduler with all components.
        policy = StaticLookaheadPolicy(
            prefetch_window=config.prefetch_window_blocks,
            eviction_cooldown_steps=config.eviction_cooldown_steps,
        )
        self._scheduler = StagehandScheduler(
            registry=self._registry,
            residency=self._residency,
            transfer_engine=self._engine,
            budget=self._budget,
            policy=policy,
            guards=self._guards,
            telemetry=self._telemetry,
            config=config,
            inference_mode=inference_mode,
        )

        # 6. VMM backend (optional, inference-only, Ampere+ GPUs).
        self._vmm: VmmManager | None = None
        self._vmm_active_blocks: dict[str, int] = {}  # block_id → block_idx
        self._vmm_block_id_to_idx: dict[str, int] = {}
        if inference_mode and vmm_is_available():
            try:
                self._vmm = VmmManager(device=0)
                # Build block_id → index mapping
                for i, entry in enumerate(self._registry.blocks_in_order()):
                    self._vmm_block_id_to_idx[entry.block_id] = i
            except Exception as e:
                log.warning("VMM init failed, using block-swap: %s", e)
                self._vmm = None

        log.info(
            "StagehandRuntime: %d blocks registered, pool=%dMB, inference=%s, vmm=%s",
            len(self._registry),
            config.pinned_pool_mb,
            inference_mode,
            self._vmm is not None,
        )

    # ── step lifecycle ────────────────────────────────────────────────

    def begin_step(self, step: int) -> None:
        """Call at the start of each training step."""
        self._scheduler.begin_step(step)

    def end_step(self) -> None:
        """Call at the end of each training step."""
        # Record pool stats and VRAM usage BEFORE scheduler.end_step()
        # because end_step() calls telemetry.end_step() which archives
        # the current metrics and sets _current = None.  Recording after
        # that point would be a no-op.
        pool_stats = self._pool.stats()
        self._telemetry.record_pool_stats(
            free=int(pool_stats["free"]),
            in_use=int(pool_stats["in_use"]),
        )
        self._telemetry.record_vram(
            used_mb=self._budget.vram_used_mb(),
            reserved_mb=self._budget.vram_reserved_mb(),
        )
        self._scheduler.end_step()

    def convert_registry_to_file_backed(self, source_path: str) -> int:
        """Convert registered blocks to file-backed mode from *source_path*."""
        converted = self._registry.convert_to_file_backed(source_path)
        self._scheduler.refresh_registry_snapshot()
        log.info(
            "StagehandRuntime: converted %d params to file-backed source (%s)",
            converted,
            source_path,
        )
        return converted

    def convert_registry_to_squareq_v2_backed(
        self, safetensors_path: str, manifest_path: str,
    ) -> int:
        """Convert registered blocks to SquareQ V2-backed mode.

        Uses a pre-quantized INT8 safetensors slab and its JSON manifest
        to back registered block parameters with quantized storage.
        """
        converted = self._registry.convert_to_squareq_v2_backed(
            safetensors_path, manifest_path,
        )
        self._scheduler.refresh_registry_snapshot()
        log.info(
            "StagehandRuntime: converted %d params to SquareQ V2-backed source (%s)",
            converted,
            safetensors_path,
        )
        return converted

    def convert_registry_to_file_backed_sharded(self, source_dir: str) -> int:
        """Convert registered blocks to file-backed mode from sharded safetensors.

        *source_dir* must contain a safetensors index JSON and the shard files.
        Each block parameter is mapped to its specific shard file, allowing
        models split across multiple files (e.g. FLUX.2-dev's 7 transformer
        shards) to stream blocks from disk without loading all weights into RAM.
        """
        converted = self._registry.convert_to_file_backed_sharded(source_dir)
        self._scheduler.refresh_registry_snapshot()
        log.info(
            "StagehandRuntime: converted %d params to sharded file-backed source (%s)",
            converted,
            source_dir,
        )
        return converted

    # ── VMM integration ─────────────────────────────────────────────

    def vmm_register_model(
        self,
        model_id: str,
        dtype_str: str = "bfloat16",
        safetensors_path: str | None = None,
    ) -> VmmModelHandle | None:
        """Register the current model's blocks with the VMM backend.

        Call after ``convert_registry_to_file_backed*`` so that block entries
        have param_layout populated.

        Parameters
        ----------
        model_id:
            Unique identifier for this model (e.g. ``"diffusion:flux-dev"``).
        dtype_str:
            Weight dtype for DLPack tensors (``"bfloat16"``, ``"float16"``, etc.).
        safetensors_path:
            If provided, weights are loaded via uncommitted mmap from this
            file — zero committed RAM.  If *None*, falls back to materializing
            CPU weight copies (uses committed RAM).

        Returns the :class:`VmmModelHandle` or *None* if VMM is not available.
        """
        if self._vmm is None:
            return None

        ordered = self._registry.blocks_in_order()
        block_sizes = [entry.size_bytes for entry in ordered]
        handle = self._vmm.register_model(model_id, block_sizes, dtype_str)

        # Register block shapes
        for i, entry in enumerate(ordered):
            total_elements = sum(
                numel for _, _, _, _, numel in (entry.param_layout or [])
            )
            if total_elements > 0:
                handle.register_block_shape(i, [total_elements])

        # Set weight source: mmap (preferred) or committed RAM (fallback)
        mmap_ok = False
        if safetensors_path is not None and safetensors_path.endswith(".safetensors"):
            # Build tensor name map: which safetensors keys belong to which block.
            # The state_dict key prefix for each block comes from the registry.
            block_tensor_map: list[list[str]] = []
            for entry in ordered:
                # Collect param names that belong to this block.
                # The registry stores dotted names relative to the block module,
                # but safetensors keys are fully-qualified from the model root.
                # Use the block's module name prefix from the model.
                block_names = []
                if entry.param_layout:
                    module = entry.module_ref()  # weakref deref → nn.Module
                    # Find the block's prefix in the model's named_modules
                    model_root = self._model
                    prefix = ""
                    if model_root is not None and module is not None:
                        for name, mod in model_root.named_modules():
                            if mod is module:
                                prefix = name + "." if name else ""
                                break
                    for param_name, _, _, _, _ in entry.param_layout:
                        block_names.append(prefix + param_name)
                block_tensor_map.append(block_names)

            try:
                handle.set_mmap_source(safetensors_path, block_tensor_map)
                mmap_ok = True
                log.info("VMM: using mmap source for %s: %s", model_id, safetensors_path)
            except Exception as e:
                log.warning("VMM: mmap source failed for %s: %s, falling back to RAM", model_id, e)

        if not mmap_ok:
            # Fallback: materialize CPU weight copies
            for i, entry in enumerate(ordered):
                module = entry.module_ref()  # weakref deref → nn.Module
                if module is None or not entry.param_layout:
                    continue
                try:
                    parts: list[torch.Tensor] = []
                    for name, shape, layout_dtype, offset_bytes, numel in entry.param_layout:
                        param = module
                        for attr in name.split("."):
                            param = getattr(param, attr)
                        parts.append(
                            param.data.to(layout_dtype).flatten().view(torch.uint8)
                        )
                    flat = torch.cat(parts).cpu()
                    handle.set_ram_weights(i, flat)
                except Exception:
                    pass

        self._vmm.activate_model(model_id)
        self._vmm_model_id = model_id
        log.info("VMM: registered model %s with %d blocks", model_id, len(block_sizes))
        return handle

    def _vmm_before_block(self, block_id: str, module: nn.Module) -> bool:
        """Try to serve a block via VMM. Returns True if successful."""
        if self._vmm is None:
            return False

        model_id = getattr(self, "_vmm_model_id", None)
        if model_id is None or model_id not in self._vmm.models:
            return False

        block_idx = self._vmm_block_id_to_idx.get(block_id)
        if block_idx is None:
            return False

        vmm_handle = self._vmm.models[model_id]
        stream = torch.cuda.current_stream().cuda_stream

        # Prefetch next block
        vmm_handle.prefetch_block(block_idx + 1)

        tensor, is_vmm = vmm_handle.get_block_tensor(block_idx, stream)
        if not is_vmm:
            return False  # watermarked — fall back to block-swap

        # Restore module params from the VMM tensor using the param layout.
        # If this fails, release the VMM handle and fall back to block-swap.
        try:
            block_entry = self._registry.get(block_id)
            if block_entry.param_layout:
                from stagehand.scheduler import _restore_params_from_tensor
                gpu_bytes = tensor.view(torch.uint8)
                _restore_params_from_tensor(module, gpu_bytes, list(block_entry.param_layout))
        except Exception as e:
            log.warning("VMM: param restore failed for %s, falling back: %s", block_id, e)
            vmm_handle.release_block(block_idx)
            return False

        self._vmm_active_blocks[block_id] = block_idx

        # Advance scheduler cursor so prefetch predictions stay aligned.
        # Also trigger prefetch-ahead — without this, mixed VMM/block-swap
        # sequences would starve the prefetcher.
        self._scheduler._cursor += 1
        self._telemetry.record_prefetch_hit()
        self._scheduler._prefetch_ahead()

        return True

    def _vmm_after_block(self, block_id: str) -> None:
        """Release VMM handle after block execution.

        Detaches module params from the VMM tensor BEFORE releasing the handle.
        Without this, param views keep the DLPack tensor alive → region refcount
        stays at 1 → the VMM allocator cannot evict/unmap the region.
        """
        block_idx = self._vmm_active_blocks.pop(block_id, None)
        if block_idx is None:
            return

        # Detach module params from VMM tensor views. This drops the DLPack
        # tensor's last Python reference (param views were the only holders),
        # triggering the DLPack deleter which decrements the VMM refcount.
        # Uses the existing _detach_params which replaces param.data with
        # empty(0) tensors, matching what the block-swap eviction path does.
        block_entry = self._registry.get(block_id)
        if block_entry is not None and block_entry.param_layout:
            module = block_entry.module_ref()
            if module is not None:
                from stagehand.scheduler import _detach_params
                _detach_params(module, list(block_entry.param_layout))

        model_id = getattr(self, "_vmm_model_id", None)
        if model_id is not None and model_id in self._vmm.models:
            self._vmm.models[model_id].release_block(block_idx)

    # ── forward/backward hooks ────────────────────────────────────────

    def _make_pre_forward_hook(self, block_id: str):  # noqa: ANN202
        """Create a pre-forward hook that ensures the block is on GPU.

        Tries VMM fast path first (inference only), falls back to block-swap.
        """
        def hook(module: nn.Module, args: tuple) -> None:
            if self._vmm is not None and self._vmm_before_block(block_id, module):
                return  # VMM served the block
            self._scheduler.before_block(block_id)
        return hook

    def _make_post_forward_hook(self, block_id: str):  # noqa: ANN202
        """Create a post-forward hook that releases the block's refcount."""
        def hook(module: nn.Module, args: tuple, output: object) -> None:
            if block_id in self._vmm_active_blocks:
                self._vmm_after_block(block_id)
                return
            out_tensor: torch.Tensor | None = None
            if isinstance(output, torch.Tensor):
                out_tensor = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    out_tensor = first
            self._scheduler.after_block(block_id, out_tensor)
        return hook

    def _make_pre_backward_hook(self, block_id: str):  # noqa: ANN202
        """Create a pre-backward hook that ensures the block is on GPU."""
        def hook(module: nn.Module, grad_output: tuple) -> None:
            self._scheduler.enter_backward_phase()
            self._scheduler.before_block(block_id)
        return hook

    def _make_post_backward_hook(self, block_id: str):  # noqa: ANN202
        """Create a post-backward hook that releases the block's refcount."""
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self._scheduler.after_block(block_id)
        return hook

    @contextmanager
    def managed_forward(self) -> Generator[None, None, None]:
        """Context manager that installs forward pre/post hooks on registered blocks."""
        handles: list[torch.utils.hooks.RemovableHook] = []
        for entry in self._registry.blocks_in_order():
            module = entry.module_ref()
            if module is None:
                continue
            h_pre = module.register_forward_pre_hook(
                self._make_pre_forward_hook(entry.block_id),
            )
            h_post = module.register_forward_hook(
                self._make_post_forward_hook(entry.block_id),
            )
            handles.append(h_pre)
            handles.append(h_post)
        try:
            yield
        finally:
            self._scheduler.exit_backward_phase()
            for h in handles:
                h.remove()

    def set_defer_backward_eviction(self, enabled: bool) -> None:
        """Control whether eager post-backward evictions are deferred."""
        self._scheduler.set_defer_backward_eviction(enabled)

    @contextmanager
    def managed_backward(self) -> Generator[None, None, None]:
        """Context manager for backward pass block management.

        Installs full_backward_pre_hook and full_backward_hook on registered
        blocks in reverse execution order.  In inference mode this is a no-op.
        """
        if self._inference_mode:
            yield
            return

        handles: list[torch.utils.hooks.RemovableHook] = []
        # Reverse order for backward pass.
        for entry in reversed(self._registry.blocks_in_order()):
            module = entry.module_ref()
            if module is None:
                continue
            h_pre = module.register_full_backward_pre_hook(
                self._make_pre_backward_hook(entry.block_id),
            )
            h_post = module.register_full_backward_hook(
                self._make_post_backward_hook(entry.block_id),
            )
            handles.append(h_pre)
            handles.append(h_post)
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ── shutdown ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: drain transfers, release pool, close telemetry.

        TODO(phase3-audit): VMM SHUTDOWN WITHOUT PARAM DETACH — vmm.shutdown()
        calls destroy() on each VmmModelHandle, which calls destroy_slab().
        If any module params still hold views into DLPack tensors from as_tensor(),
        destroy_slab() will fail with SlabNotEmpty. Must iterate all registered
        blocks and _detach_params() before calling vmm.shutdown().
        Same issue in shutdown_keep_pool() below.
        """
        if self._vmm is not None:
            self._vmm.shutdown()
            self._vmm = None
        if _DEBUG_STAGEHAND_SHUTDOWN:
            print("[stagehand/debug] runtime.shutdown: engine drain start", flush=True)
        self._engine.drain()
        if _DEBUG_STAGEHAND_SHUTDOWN:
            print("[stagehand/debug] runtime.shutdown: engine drain done", flush=True)
        self._scheduler.close()
        if _DEBUG_STAGEHAND_SHUTDOWN:
            print("[stagehand/debug] runtime.shutdown: scheduler close done", flush=True)
        self._pool.shutdown()
        if _DEBUG_STAGEHAND_SHUTDOWN:
            print("[stagehand/debug] runtime.shutdown: pool shutdown done", flush=True)
        self._telemetry.close()
        if _DEBUG_STAGEHAND_SHUTDOWN:
            print("[stagehand/debug] runtime.shutdown: telemetry close done", flush=True)

    def shutdown_keep_pool(self) -> PinnedPool:
        """Shut down runtime but return the pool for reuse by another runtime.

        Drains in-flight transfers and closes telemetry, but does NOT
        shut down the pinned pool.  The caller is responsible for either
        passing the pool to a new ``StagehandRuntime`` or calling
        ``pool.shutdown()`` when done.

        TODO(phase3-audit): Same VMM param detach issue as shutdown() — see above.
        """
        if self._vmm is not None:
            self._vmm.shutdown()
            self._vmm = None
        self._engine.drain()
        self._scheduler.close()
        self._telemetry.close()
        pool = self._pool
        return pool

    # ── properties ────────────────────────────────────────────────────

    @property
    def telemetry(self) -> StagehandTelemetry:
        """Access the telemetry recorder."""
        return self._telemetry

    @property
    def stats(self) -> dict:
        """Quick status snapshot."""
        return {
            "pool": self._pool.stats(),
            "telemetry": {
                "hit_rate": self._telemetry.hit_rate(),
                "mean_stall_ms": self._telemetry.mean_stall_ms(),
            },
        }


# ── layer-mode top-level API ──────────────────────────────────────────────


def layer(
    model: nn.Module,
    *,
    vram_budget: str | int | None = None,
    ram_budget: str | int | None = None,
    prefetch_k: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    inference_mode: bool = False,
    telemetry: bool = True,
    pool: PinnedPool | None = None,
) -> nn.Module:
    """One-line layer-mode API.  Works on any model.

    Wraps individual ``nn.Linear``, ``nn.Conv2d``, ``nn.Embedding`` modules
    and manages CPU/GPU transfer through a bounded pinned pool.

    Returns the same model with hooks installed and a ``_stagehand_layer_runtime``
    attribute pointing to the :class:`LayerRuntime` instance.
    """
    runtime = LayerRuntime(
        model,
        vram_budget=vram_budget,
        ram_budget=ram_budget,
        prefetch_k=prefetch_k,
        dtype=dtype,
        inference_mode=inference_mode,
        telemetry=telemetry,
        pool=pool,
    )
    model._stagehand_layer_runtime = runtime  # type: ignore[attr-defined]
    return model


def wrap(model: nn.Module, **kwargs: object) -> nn.Module:
    """Auto-detect best mode.  Currently calls :func:`layer`."""
    return layer(model, **kwargs)  # type: ignore[arg-type]


class Runtime:
    """Optional config holder for explicit configuration.

    Usage::

        rt = stagehand.Runtime(vram_budget="4GB", prefetch_k=5)
        model = rt.layer(model)
    """

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs

    def layer(self, model: nn.Module) -> nn.Module:
        return layer(model, **self._kwargs)  # type: ignore[arg-type]

    def wrap(self, model: nn.Module) -> nn.Module:
        return wrap(model, **self._kwargs)
