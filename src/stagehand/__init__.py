"""Stagehand — bounded block-swapping runtime for Serenity.

Top-level package that re-exports all public types from every module
and provides the :class:`StagehandRuntime` integration API.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
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
from stagehand.residency import BlockState, ResidencyEntry, ResidencyMap, ResidentPriority
from stagehand.scheduler import StaticLookaheadPolicy, StagehandScheduler
from stagehand.telemetry import StagehandTelemetry, StepMetrics
from stagehand.transfer import AsyncTransferEngine, TransferHandle

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
    "ResidentPriority",
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

__version__ = "0.1.0"

log = logging.getLogger(__name__)


@dataclass
class _ReservationEntry:
    """Internal tracking for reserve_for_resident."""

    runtime: object  # StagehandRuntime (forward ref)
    size_bytes: int
    block_ids: frozenset[str]
    priority: ResidentPriority


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

        log.info(
            "StagehandRuntime: %d blocks registered, pool=%dMB, inference=%s",
            len(self._registry),
            config.pinned_pool_mb,
            inference_mode,
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

    # ── forward/backward hooks ────────────────────────────────────────

    def _make_pre_forward_hook(self, block_id: str):  # noqa: ANN202
        """Create a pre-forward hook that ensures the block is on GPU."""
        def hook(module: nn.Module, args: tuple) -> None:
            self._scheduler.before_block(block_id)
        return hook

    def _make_post_forward_hook(self, block_id: str):  # noqa: ANN202
        """Create a post-forward hook that releases the block's refcount."""
        def hook(module: nn.Module, args: tuple, output: object) -> None:
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
            for h in handles:
                h.remove()

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

    # ── residency protection ─────────────────────────────────────────

    @contextmanager
    def keep_resident(self, model_or_runtime: StagehandRuntime | None = None) -> Generator[None, None, None]:
        """Suppress eviction of blocks for the duration of the context.

        Blocks with refcount 0 will not be selected as eviction candidates
        while this context is active. On context exit, protection is lifted.
        Explicit eviction must be called separately if the model should be
        unloaded.

        If *model_or_runtime* is another ``StagehandRuntime``, its block IDs
        are used. If ``None``, protects this runtime's own blocks.

        Example::

            with runtime.keep_resident():
                result1 = model.generate(...)
                result2 = model.encode(...)
            # blocks are now normal eviction candidates again
        """
        target = model_or_runtime if model_or_runtime is not None else self
        block_ids = frozenset(
            entry.block_id for entry in target._registry.blocks_in_order()
        )
        if not block_ids:
            log.warning(
                "keep_resident: no blocks found. Model may not be registered "
                "with Stagehand."
            )
            yield
            return

        self._scheduler.protect_blocks(block_ids)
        log.debug("keep_resident: protecting %d blocks", len(block_ids))
        try:
            yield
        finally:
            self._scheduler.unprotect_blocks(block_ids)
            log.debug("keep_resident: released protection for %d blocks", len(block_ids))

    def reserve_for_resident(
        self,
        model_or_runtime: StagehandRuntime | None = None,
        priority: ResidentPriority = ResidentPriority.PRIMARY,
    ) -> None:
        """Reserve VRAM headroom for a model that must stay resident.

        Effects:
        - All blocks are marked with the given priority
        - BudgetManager subtracts model's size from "available for guests"
        - Scheduler will not select PRIMARY blocks as eviction candidates
          while reservation is active

        If *model_or_runtime* is ``None``, reserves this runtime's own blocks.
        """
        target = model_or_runtime if model_or_runtime is not None else self
        block_ids = frozenset(
            entry.block_id for entry in target._registry.blocks_in_order()
        )
        size_bytes = sum(
            entry.size_bytes for entry in target._registry.blocks_in_order()
        )

        if not hasattr(self, "_reservations"):
            self._reservations: dict[int, _ReservationEntry] = {}

        self._reservations[id(target)] = _ReservationEntry(
            runtime=target,
            size_bytes=size_bytes,
            block_ids=block_ids,
            priority=priority,
        )

        if priority == ResidentPriority.PRIMARY:
            self._scheduler.protect_blocks(block_ids)
        self._budget.reserve_bytes(size_bytes, label=repr(target))

        log.info(
            "reserve_for_resident: reserved %.2fGB (%d blocks) at priority=%s",
            size_bytes / 1e9,
            len(block_ids),
            priority.value,
        )

    def release_reservation(
        self,
        model_or_runtime: StagehandRuntime | None = None,
    ) -> None:
        """Release a reservation, returning headroom to the guest pool."""
        target = model_or_runtime if model_or_runtime is not None else self
        if not hasattr(self, "_reservations"):
            return
        reservation = self._reservations.pop(id(target), None)
        if reservation is None:
            return
        if reservation.priority == ResidentPriority.PRIMARY:
            self._scheduler.unprotect_blocks(reservation.block_ids)
        self._budget.release_reserved_bytes(reservation.size_bytes)
        log.info(
            "release_reservation: released %.2fGB reservation",
            reservation.size_bytes / 1e9,
        )

    @contextmanager
    def as_guest(self, guest_runtime: StagehandRuntime) -> Generator[None, None, None]:
        """Load a short-lived guest model, evaluated against guest headroom only.

        If insufficient guest headroom exists, the prefetch window is reduced.
        Protected PRIMARY blocks are never evicted to make room for guests.
        On context exit, the guest reservation is released.

        Example::

            with conductor.as_guest(spatial_upscaler):
                result = spatial_upscaler.run(latent)
            # guest_runtime evicted on context exit
        """
        size_bytes = sum(
            entry.size_bytes for entry in guest_runtime._registry.blocks_in_order()
        )

        prev_window = None
        if not self._budget.can_guest_allocate(size_bytes):
            prev_window = self._scheduler._policy.prefetch_window
            self._scheduler._policy.prefetch_window = 1
            log.warning(
                "as_guest: tight headroom for guest (%.2fGB), reduced prefetch window",
                size_bytes / 1e9,
            )

        self.reserve_for_resident(guest_runtime, priority=ResidentPriority.GUEST)
        try:
            yield
        finally:
            self.release_reservation(guest_runtime)
            if prev_window is not None:
                self._scheduler._policy.prefetch_window = prev_window

    # ── shutdown ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: drain transfers, release pool, close telemetry."""
        self._engine.drain()
        self._scheduler.close()
        self._pool.shutdown()
        self._telemetry.close()

    def shutdown_keep_pool(self) -> PinnedPool:
        """Shut down runtime but return the pool for reuse by another runtime.

        Drains in-flight transfers and closes telemetry, but does NOT
        shut down the pinned pool.  The caller is responsible for either
        passing the pool to a new ``StagehandRuntime`` or calling
        ``pool.shutdown()`` when done.
        """
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
