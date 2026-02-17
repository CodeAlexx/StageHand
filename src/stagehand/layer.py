"""Layer-mode runtime for the stagehand block-swapping engine.

Wraps individual ``nn.Linear``, ``nn.Conv2d``, and ``nn.Embedding``
modules and routes all transfers through the same bounded runtime used
by block mode.  One line, zero config, works on ANY model::

    model = stagehand.layer(model)

Two-phase lifecycle:

* **Trace** (step 0): Walk-order ``exec_order``, ``prefetch_window=0``.
  Forward hooks record actual call order into ``_trace_order``.
* **Scheduled** (step 1+): Registry rebuilt with traced order and real
  ``prefetch_window``.  k-lookahead prefetch hides transfer latency.

Auto-step detection: When the first-in-traced-order layer fires again,
the previous step ends and a new step begins.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch
from torch import nn

from stagehand.budget import BudgetManager
from stagehand.config import StagehandConfig
from stagehand.guards import NumericGuard
from stagehand.pool import PinnedPool
from stagehand.registry import BlockRegistry
from stagehand.residency import ResidencyMap
from stagehand.scheduler import StaticLookaheadPolicy, StagehandScheduler
from stagehand.telemetry import StagehandTelemetry
from stagehand.transfer import AsyncTransferEngine

__all__ = ["LayerRuntime"]

log = logging.getLogger(__name__)

# Types eligible for layer-mode wrapping.
LAYER_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d, nn.Embedding)


# ── helpers ──────────────────────────────────────────────────────────────


def _discover_layers(
    model: nn.Module,
    target_types: tuple[type[nn.Module], ...] = LAYER_TYPES,
) -> list[tuple[str, nn.Module]]:
    """Walk ``model.named_modules()`` and return leaf modules of target types.

    Skips shared (aliased) modules via ``id()`` tracking.
    """
    seen_ids: set[int] = set()
    result: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not isinstance(module, target_types):
            continue
        # Must have own parameters.
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        mid = id(module)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        result.append((name, module))
    return result


def _parse_budget(value: str | int | None) -> int | None:
    """Parse a budget string like ``"4GB"`` or ``"512MB"`` to bytes.

    Returns *None* if *value* is None (auto-detect).
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    v = value.strip().upper()
    if v.endswith("GB"):
        return int(float(v[:-2]) * 1024 * 1024 * 1024)
    if v.endswith("MB"):
        return int(float(v[:-2]) * 1024 * 1024)
    if v.endswith("B"):
        return int(v[:-1])
    return int(v)


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _auto_pool_config(
    layers: list[tuple[str, nn.Module]],
    dtype: torch.dtype,
    prefetch_k: int,
    vram_budget: int | None,
    ram_budget: int | None,
) -> StagehandConfig:
    """Derive ``StagehandConfig`` from discovered layers.

    * Slab size: ``ceil(max_layer_bytes / 1 MiB)`` rounded to next power of 2 MiB.
    * Pool size: ``slab_mb * (prefetch_k + 4)`` slabs minimum.
    * VRAM: 80% high / 60% low of detected VRAM (or user budget).
    """
    max_layer_bytes = 0
    for _name, module in layers:
        size = 0
        for p in module.parameters(recurse=False):
            size += p.numel() * dtype.itemsize
        if size > max_layer_bytes:
            max_layer_bytes = size

    # Slab sizing: round to power of 2 MiB, minimum 1 MiB.
    slab_mb = max(1, _next_power_of_two(math.ceil(max_layer_bytes / (1024 * 1024))))

    # Pool sizing: enough slabs for prefetch + active + margin.
    num_slabs = prefetch_k + 4
    total_mb = slab_mb * num_slabs

    # RAM cap.
    if ram_budget is not None:
        ram_mb = ram_budget // (1024 * 1024)
        if total_mb > ram_mb:
            total_mb = max(slab_mb, (ram_mb // slab_mb) * slab_mb)

    # VRAM watermarks.
    if vram_budget is not None:
        vram_mb = vram_budget // (1024 * 1024)
        high_mb = int(vram_mb * 0.80)
        low_mb = int(vram_mb * 0.60)
    elif torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        vram_mb = total_vram // (1024 * 1024)
        high_mb = int(vram_mb * 0.80)
        low_mb = int(vram_mb * 0.60)
    else:
        high_mb = 22000
        low_mb = 18000

    # Ensure high > low.
    if high_mb <= low_mb:
        high_mb = low_mb + 1

    return StagehandConfig(
        pinned_pool_mb=total_mb,
        pinned_slab_mb=slab_mb,
        vram_high_watermark_mb=high_mb,
        vram_low_watermark_mb=low_mb,
        prefetch_window_blocks=prefetch_k,
        eviction_cooldown_steps=0,
        nan_inf_check=False,
        strict_bf16=False,
        fail_on_dtype_promotion=False,
        telemetry_enabled=True,
        telemetry_interval_steps=10,
    )


# ── runtime ──────────────────────────────────────────────────────────────


class LayerRuntime:
    """Per-module CPU offloading runtime.

    Discovers leaf modules, learns execution order from a trace pass (step 0),
    then drives the existing ``StagehandScheduler`` with correctly-ordered entries.

    Parameters
    ----------
    model:
        Any ``nn.Module`` to wrap.
    vram_budget:
        VRAM limit as bytes or string (``"4GB"``).  None = auto-detect.
    ram_budget:
        RAM limit for the pinned pool.  None = auto-size.
    prefetch_k:
        Number of layers to prefetch ahead after trace.
    dtype:
        Storage dtype for parameter transfers.
    inference_mode:
        If True, no backward hooks; eviction never saves back.
    telemetry:
        Enable telemetry recording.
    pool:
        Optional pre-existing :class:`PinnedPool` to reuse.
    """

    def __init__(
        self,
        model: nn.Module,
        vram_budget: str | int | None = None,
        ram_budget: str | int | None = None,
        prefetch_k: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        inference_mode: bool = False,
        telemetry: bool = True,
        pool: PinnedPool | None = None,
    ) -> None:
        self._model = model
        self._dtype = dtype
        self._prefetch_k = prefetch_k
        self._inference_mode = inference_mode

        # 1. Discover layers.
        layers = _discover_layers(model)
        if not layers:
            raise ValueError(
                "No eligible layers found (nn.Linear, nn.Conv2d, nn.Embedding). "
                "Model must contain at least one parameterized layer."
            )

        self._layer_map: dict[str, nn.Module] = {name: mod for name, mod in layers}
        self._all_layer_names: list[str] = [name for name, _ in layers]

        # 2. Auto-configure pool.
        vram_bytes = _parse_budget(vram_budget)
        ram_bytes = _parse_budget(ram_budget)
        self._config = _auto_pool_config(
            layers, dtype, prefetch_k, vram_bytes, ram_bytes,
        )
        self._config.telemetry_enabled = telemetry

        # 3. Create shared infrastructure.
        if pool is not None:
            self._pool = pool
        else:
            self._pool = PinnedPool(
                total_mb=self._config.pinned_pool_mb,
                slab_mb=self._config.pinned_slab_mb,
            )

        copy_stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available():
            copy_stream = torch.cuda.Stream(priority=self._config.copy_stream_priority)
        self._engine = AsyncTransferEngine(
            pool=self._pool,
            max_inflight=self._config.max_inflight_transfers,
            copy_stream=copy_stream,
        )

        self._budget = BudgetManager(
            high_watermark_mb=self._config.vram_high_watermark_mb,
            low_watermark_mb=self._config.vram_low_watermark_mb,
        )

        self._guards: NumericGuard | None = None
        self._telemetry_inst = StagehandTelemetry(
            enabled=self._config.telemetry_enabled,
            interval_steps=self._config.telemetry_interval_steps,
            output_file=None,
        )

        # 4. Build initial registry from walk order (prefetch_window=0 for trace).
        self._registry = BlockRegistry()
        self._registry.build_from_module_list(layers, "layer", dtype)
        self._registry.validate(
            pool_capacity_bytes=self._pool.num_slabs * self._pool.slab_bytes,
        )

        # 5. Create residency + scheduler (trace mode: prefetch=0).
        self._residency = ResidencyMap(self._registry)
        trace_policy = StaticLookaheadPolicy(
            prefetch_window=0, eviction_cooldown_steps=0,
        )
        self._scheduler = StagehandScheduler(
            registry=self._registry,
            residency=self._residency,
            transfer_engine=self._engine,
            budget=self._budget,
            policy=trace_policy,
            guards=self._guards,
            telemetry=self._telemetry_inst,
            config=self._config,
            inference_mode=inference_mode,
        )

        # 6. Move non-managed parameters to GPU.
        # Managed layers (Linear, Conv2d, Embedding) are offloaded by
        # the scheduler.  Everything else (LayerNorm, biases in container
        # modules, etc.) is tiny and should live on GPU permanently.
        if torch.cuda.is_available():
            managed_ids = {id(m) for m in self._layer_map.values()}
            for module in model.modules():
                if id(module) in managed_ids:
                    continue
                for p in module.parameters(recurse=False):
                    p.data = p.data.to("cuda", dtype=dtype)
                for name, buf in module.named_buffers(recurse=False):
                    setattr(module, name, buf.to("cuda", dtype=dtype))

        # 7. Trace state.
        self._trace_order: list[str] = []
        self._trace_seen: set[str] = set()
        self._step: int = 0
        self._forward_started: bool = False
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._mode: str = "trace"

        # 8. Install trace hooks.
        self._install_trace_hooks()

        log.info(
            "LayerRuntime: %d layers discovered, pool=%dMB (%dMB slabs), "
            "prefetch_k=%d, mode=trace",
            len(layers),
            self._config.pinned_pool_mb,
            self._config.pinned_slab_mb,
            prefetch_k,
        )

    # ── hook installation ────────────────────────────────────────────

    def _install_trace_hooks(self) -> None:
        """Install forward/backward hooks for trace mode (step 0)."""
        for entry in self._registry.blocks_in_order():
            module = entry.module_ref()
            if module is None:
                continue
            layer_id = entry.block_id

            h_pre = module.register_forward_pre_hook(
                self._make_trace_pre_hook(layer_id),
            )
            h_post = module.register_forward_hook(
                self._make_post_hook(layer_id),
            )
            self._hook_handles.append(h_pre)
            self._hook_handles.append(h_post)

            if not self._inference_mode:
                h_bwd_pre = module.register_full_backward_pre_hook(
                    self._make_bwd_pre_hook(layer_id),
                )
                h_bwd_post = module.register_full_backward_hook(
                    self._make_bwd_post_hook(layer_id),
                )
                self._hook_handles.append(h_bwd_pre)
                self._hook_handles.append(h_bwd_post)

    def _install_scheduled_hooks(self) -> None:
        """Install forward/backward hooks for scheduled mode (step 1+)."""
        # Forward hooks in traced order.
        for layer_id in self._trace_order:
            module = self._layer_map.get(layer_id)
            if module is None:
                continue

            h_pre = module.register_forward_pre_hook(
                self._make_scheduled_pre_hook(layer_id),
            )
            h_post = module.register_forward_hook(
                self._make_post_hook(layer_id),
            )
            self._hook_handles.append(h_pre)
            self._hook_handles.append(h_post)

        # Backward hooks in reversed traced order (training only).
        if not self._inference_mode:
            for layer_id in reversed(self._trace_order):
                module = self._layer_map.get(layer_id)
                if module is None:
                    continue

                h_bwd_pre = module.register_full_backward_pre_hook(
                    self._make_bwd_pre_hook(layer_id),
                )
                h_bwd_post = module.register_full_backward_hook(
                    self._make_bwd_post_hook(layer_id),
                )
                self._hook_handles.append(h_bwd_pre)
                self._hook_handles.append(h_bwd_post)

    # ── hook factories ───────────────────────────────────────────────

    def _make_trace_pre_hook(self, layer_id: str):  # noqa: ANN202
        """Forward pre-hook for trace mode: records order + auto-step."""
        def hook(module: nn.Module, args: tuple) -> None:
            # Auto-step detection: if the first traced layer fires again,
            # the trace is complete and we rebuild.
            if (
                self._trace_order
                and layer_id == self._trace_order[0]
                and self._forward_started
            ):
                self._complete_trace()
                # _complete_trace installed new scheduled hooks and called
                # begin_step(1).  We still need before_block for this layer
                # because the new scheduled pre-hook won't fire for this
                # call (hooks were just installed mid-call).
                self._scheduler.before_block(layer_id)
                return

            # Record trace order (first occurrence only).
            if layer_id not in self._trace_seen:
                self._trace_order.append(layer_id)
                self._trace_seen.add(layer_id)

            # Begin step 0 on first hook ever.
            if not self._forward_started:
                self._scheduler.begin_step(self._step)
                self._forward_started = True

            self._scheduler.before_block(layer_id)

        return hook

    def _make_scheduled_pre_hook(self, layer_id: str):  # noqa: ANN202
        """Forward pre-hook for scheduled mode: auto-step + before_block."""
        def hook(module: nn.Module, args: tuple) -> None:
            # Auto-step: when first layer fires again, end previous step.
            if layer_id == self._trace_order[0] and self._forward_started:
                self._record_pool_stats()
                self._scheduler.end_step()
                self._step += 1
                self._scheduler.begin_step(self._step)

            if not self._forward_started:
                self._scheduler.begin_step(self._step)
                self._forward_started = True

            self._scheduler.before_block(layer_id)

        return hook

    def _make_post_hook(self, layer_id: str):  # noqa: ANN202
        """Forward post-hook: after_block with output extraction."""
        def hook(module: nn.Module, args: tuple, output: object) -> None:
            out_tensor: torch.Tensor | None = None
            if isinstance(output, torch.Tensor):
                out_tensor = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    out_tensor = first
            self._scheduler.after_block(layer_id, out_tensor)

        return hook

    def _make_bwd_pre_hook(self, layer_id: str):  # noqa: ANN202
        """Backward pre-hook: ensure layer is on GPU."""
        def hook(module: nn.Module, grad_output: tuple) -> None:
            self._scheduler.before_block(layer_id)

        return hook

    def _make_bwd_post_hook(self, layer_id: str):  # noqa: ANN202
        """Backward post-hook: release layer."""
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self._scheduler.after_block(layer_id)

        return hook

    # ── trace completion / rebuild ───────────────────────────────────

    def _complete_trace(self) -> None:
        """End step 0, rebuild registry with traced order, begin step 1."""
        # 1. End step 0.
        self._record_pool_stats()
        self._scheduler.end_step()

        # 2. Remove all trace hooks.
        self._remove_hooks()

        # 3. Rebuild registry with traced execution order.
        traced_modules = [(lid, self._layer_map[lid]) for lid in self._trace_order]
        self._registry = BlockRegistry()
        self._registry.build_from_module_list(traced_modules, "layer", self._dtype)
        self._registry.validate(
            pool_capacity_bytes=self._pool.num_slabs * self._pool.slab_bytes,
        )

        # 4. New residency, policy, scheduler (pool/engine/budget/guards/telemetry reused).
        self._residency = ResidencyMap(self._registry)
        policy = StaticLookaheadPolicy(
            prefetch_window=self._prefetch_k,
            eviction_cooldown_steps=0,
        )
        self._scheduler = StagehandScheduler(
            registry=self._registry,
            residency=self._residency,
            transfer_engine=self._engine,
            budget=self._budget,
            policy=policy,
            guards=self._guards,
            telemetry=self._telemetry_inst,
            config=self._config,
            inference_mode=self._inference_mode,
        )

        # 5. Install scheduled hooks.
        self._install_scheduled_hooks()

        # 6. Begin step 1.
        self._step += 1
        self._scheduler.begin_step(self._step)
        self._forward_started = True
        self._mode = "scheduled"

        log.info(
            "LayerRuntime: trace complete (%d layers), rebuilding with prefetch_k=%d",
            len(self._trace_order),
            self._prefetch_k,
        )

    # ── hook management ──────────────────────────────────────────────

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── telemetry helpers ────────────────────────────────────────────

    def _record_pool_stats(self) -> None:
        """Record pool and VRAM stats to telemetry."""
        pool_stats = self._pool.stats()
        self._telemetry_inst.record_pool_stats(
            free=int(pool_stats["free"]),
            in_use=int(pool_stats["in_use"]),
        )
        self._telemetry_inst.record_vram(
            used_mb=self._budget.vram_used_mb(),
            reserved_mb=self._budget.vram_reserved_mb(),
        )

    # ── shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: remove hooks, drain transfers, close all."""
        if self._mode == "shutdown":
            return
        self._remove_hooks()
        self._engine.drain()
        self._scheduler.close()
        self._pool.shutdown()
        self._telemetry_inst.close()
        self._mode = "shutdown"

    def shutdown_keep_pool(self) -> PinnedPool:
        """Shut down but return the pool for reuse by another runtime."""
        if self._mode == "shutdown":
            raise RuntimeError("Already shut down")
        self._remove_hooks()
        self._engine.drain()
        self._scheduler.close()
        self._telemetry_inst.close()
        self._mode = "shutdown"
        return self._pool

    # ── properties ───────────────────────────────────────────────────

    @property
    def telemetry(self) -> StagehandTelemetry:
        return self._telemetry_inst

    @property
    def stats(self) -> dict:
        return {
            "pool": self._pool.stats(),
            "telemetry": {
                "hit_rate": self._telemetry_inst.hit_rate(),
                "mean_stall_ms": self._telemetry_inst.mean_stall_ms(),
            },
        }

    @property
    def traced(self) -> bool:
        """True if trace is complete and scheduler uses real prefetch."""
        return self._mode == "scheduled"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def num_layers(self) -> int:
        return len(self._all_layer_names)

    @property
    def trace_order(self) -> list[str]:
        """Layer names in traced execution order (empty before trace completes)."""
        return list(self._trace_order)

    @property
    def step(self) -> int:
        return self._step
