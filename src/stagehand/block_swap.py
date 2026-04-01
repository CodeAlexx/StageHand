"""BlockSwapScheduler -- fast paired block swap for LoRA training.

Standalone module with zero imports from other stagehand modules.
Designed for fixed-order forward (0->N) / backward (N->0) execution
where only frozen base weights swap and mutable LoRA params stay GPU-resident.
"""
from __future__ import annotations

import gc
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clean_memory_on_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _weighs_to_device(layer: nn.Module, device: torch.device) -> None:
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            if module.__class__.__name__.endswith("Linear"):
                module.weight.data = module.weight.data.to(
                    device, non_blocking=device.type != "cpu",
                )


def _swap_weight_devices_no_cuda(
    device: torch.device,
    block_to_cpu: nn.Module,
    block_to_cuda: nn.Module,
) -> None:
    # Match modules by name (consistent with CUDA path, handles asymmetric
    # module trees like SD3 where zip(modules(), modules()) breaks).
    modules_to_cpu = {
        name: module for name, module in block_to_cpu.named_modules()
    }

    swap_jobs: list[tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []
    for name, mod_cuda in block_to_cuda.named_modules():
        if not hasattr(mod_cuda, "weight") or mod_cuda.weight is None:
            continue
        mod_cpu = modules_to_cpu.get(name)
        if mod_cpu is None or mod_cpu.weight is None:
            continue
        if mod_cpu.weight.data.shape != mod_cuda.weight.data.shape:
            continue
        swap_jobs.append((
            mod_cpu, mod_cuda,
            mod_cpu.weight.data, mod_cuda.weight.data,
        ))

    # Pass 1: GPU -> CPU (copy=True ensures a new tensor even when already on CPU)
    for module_to_cpu, _, cuda_data_view, _ in swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True, copy=True)
    _synchronize_device(device)

    # Pass 2: CPU -> GPU
    for _, module_to_cuda, cuda_data_view, cpu_data_view in swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view
    _synchronize_device(device)


# ---------------------------------------------------------------------------
# Timer helper for debug instrumentation
# ---------------------------------------------------------------------------

class _Timer:
    """Lightweight per-section timer.  Zero-cost when disabled."""

    __slots__ = ("enabled", "sections", "_current_section", "_start")

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.sections: dict[str, float] = defaultdict(float)
        self._current_section: Optional[str] = None
        self._start: float = 0.0

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] += time.perf_counter() - start

    @property
    def total(self) -> float:
        return sum(self.sections.values())


# ---------------------------------------------------------------------------
# BlockSwapOffloader  (base)
# ---------------------------------------------------------------------------

class BlockSwapOffloader:
    """Base class providing weight-swap mechanics and async submission."""

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ) -> None:
        if blocks_to_swap > num_blocks:
            raise ValueError(
                f"blocks_to_swap ({blocks_to_swap}) exceeds num_blocks ({num_blocks})"
            )
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        self.debug = debug
        if not self.debug:
            if os.environ.get("STAGEHAND_BLOCK_SWAP_DEBUG", "") == "1":
                self.debug = True

        self.debug_block_count: int = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures: dict[int, Future] = {}
        self.cuda_available = device.type == "cuda"
        self.stream: Optional[torch.cuda.Stream] = None
        if self.cuda_available:
            self.stream = torch.cuda.Stream(device=device)

        # Staging buffers -- lazily allocated on first swap
        self.staging_buffer_a: Optional[list[torch.Tensor]] = None
        self.staging_buffer_b: Optional[list[torch.Tensor]] = None
        self.pinned_buffer: Optional[list[torch.Tensor]] = None

    # -- CUDA swap (main hot path) -----------------------------------------

    def swap_weight_devices_cuda(
        self,
        device: torch.device,
        block_to_cpu: nn.Module,
        block_to_cuda: nn.Module,
    ) -> Optional[torch.cuda.Event]:
        # Match modules by name (handles asymmetric module trees like SD3)
        modules_to_cpu = {
            name: module for name, module in block_to_cpu.named_modules()
        }

        swap_jobs: list[tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []
        for name, module_to_cuda in block_to_cuda.named_modules():
            if not hasattr(module_to_cuda, "weight") or module_to_cuda.weight is None:
                continue
            if not module_to_cuda.__class__.__name__.endswith("Linear"):
                continue
            module_to_cpu = modules_to_cpu.get(name)
            if module_to_cpu is not None and module_to_cpu.weight.data.shape == module_to_cuda.weight.data.shape:
                swap_jobs.append((
                    module_to_cpu, module_to_cuda,
                    module_to_cpu.weight.data, module_to_cuda.weight.data,
                ))
            else:
                # No matching CPU counterpart or shape mismatch -- move directly
                if module_to_cuda.weight.data.device != device:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        if not swap_jobs:
            return None

        # Debug timing
        self.debug_block_count += 1
        enable_timing = self.debug and (self.debug_block_count % 10 == 0)
        timer = _Timer(enabled=enable_timing)

        # CRITICAL: wait for outgoing block's forward/backward to complete
        with timer.section("sync_current_stream"):
            torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            return self._swap_non_pinned(device, swap_jobs, timer)
        else:
            return self._swap_pinned(device, swap_jobs, timer)

    def _swap_non_pinned(
        self,
        device: torch.device,
        swap_jobs: list[tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]],
        timer: _Timer,
    ) -> Optional[torch.cuda.Event]:
        stream = self.stream
        assert stream is not None

        with torch.cuda.stream(stream):
            # Lazy init or reallocate staging buffers when shapes change
            needs_alloc = self.staging_buffer_a is None
            if not needs_alloc:
                # Check if shapes still match (heterogeneous blocks)
                if len(self.staging_buffer_a) != len(swap_jobs):
                    needs_alloc = True
                else:
                    for sbuf, (_, _, cuda_view, _) in zip(self.staging_buffer_a, swap_jobs):
                        if sbuf.shape != cuda_view.shape or sbuf.dtype != cuda_view.dtype:
                            needs_alloc = True
                            break
            if needs_alloc:
                with timer.section("alloc_staging"):
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in swap_jobs
                    ]

            event_b: Optional[torch.cuda.Event] = None

            for i, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in enumerate(swap_jobs):
                sbuf_a = self.staging_buffer_a[i]
                sbuf_b = self.staging_buffer_b[i]

                # GPU -> staging A (async)
                with timer.section("gpu_to_staging_a"):
                    sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                event_a = stream.record_event()

                # Wait for previous staging B -> GPU to finish
                if event_b is not None:
                    with timer.section("wait_event_b"):
                        event_b.synchronize()

                # CPU -> staging B (synchronous -- bottleneck, overlaps with gpu->staging_a)
                with timer.section("cpu_to_staging_b"):
                    sbuf_b.copy_(module_to_cuda.weight.data)

                # Wait for GPU -> staging A to finish so cuda_data_view can be reused
                with timer.section("wait_event_a"):
                    event_a.synchronize()

                # Staging B -> GPU (async)
                with timer.section("staging_b_to_gpu"):
                    cuda_data_view.copy_(sbuf_b, non_blocking=True)
                event_b = stream.record_event()

                # Staging A -> CPU (synchronous -- other bottleneck, overlaps with staging_b->gpu)
                with timer.section("staging_a_to_cpu"):
                    cpu_data_view.copy_(sbuf_a)

                # Update weight references
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

        if timer.enabled:
            print(
                f"[{self.block_type}] Weight swap timing at {self.debug_block_count}:",
                flush=True,
            )
            for sec_name, sec_time in timer.sections.items():
                print(f"  {sec_name}: {sec_time * 1000:.2f}ms", flush=True)
            print(f"  TOTAL: {timer.total * 1000:.2f}ms", flush=True)

        return event_b

    def _swap_pinned(
        self,
        device: torch.device,
        swap_jobs: list[tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]],
        timer: _Timer,
    ) -> Optional[torch.cuda.Event]:
        stream = self.stream
        assert stream is not None

        # Lazy init or reallocate pinned buffer when shapes change
        needs_alloc = self.pinned_buffer is None
        if not needs_alloc:
            if len(self.pinned_buffer) != len(swap_jobs):
                needs_alloc = True
            else:
                for pbuf, (_, _, cuda_view, _) in zip(self.pinned_buffer, swap_jobs):
                    if pbuf.shape != cuda_view.shape or pbuf.dtype != cuda_view.dtype:
                        needs_alloc = True
                        break
        if needs_alloc:
            with timer.section("alloc_pinned"):
                with torch.cuda.stream(stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in swap_jobs
                    ]
                    stream.synchronize()

        released_pinned_buffer: list[torch.Tensor] = []
        events: list[torch.cuda.Event] = []

        # Pass 1: GPU -> pinned
        with timer.section("gpu_to_pinned"):
            with torch.cuda.stream(stream):
                for i, (_, _, cuda_data_view, _) in enumerate(swap_jobs):
                    pinned_buf = self.pinned_buffer[i]
                    pinned_buf.copy_(cuda_data_view, non_blocking=True)
                    events.append(stream.record_event())

        # Pass 2: CPU -> GPU (reusing cuda_data_view)
        with timer.section("cpu_to_gpu"):
            with torch.cuda.stream(stream):
                for i, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in enumerate(swap_jobs):
                    stream.wait_event(events[i])  # wait for cuda_data_view to be free
                    cuda_data_view.copy_(cpu_data_view, non_blocking=True)

                    # Update references
                    module_to_cuda.weight.data = cuda_data_view
                    module_to_cpu.weight.data = self.pinned_buffer[i]  # old pinned becomes CPU storage
                    released_pinned_buffer.append(cpu_data_view)

        # Recycle released buffers for next swap
        if released_pinned_buffer and not released_pinned_buffer[0].is_pinned():
            # First time: released buffers are not pinned, allocate new ones
            with timer.section("alloc_replacement_pinned"):
                with torch.cuda.stream(stream):
                    self.pinned_buffer = [
                        torch.empty_like(t, device="cpu").pin_memory(device=device)
                        for t in released_pinned_buffer
                    ]
        else:
            self.pinned_buffer = released_pinned_buffer

        final_event = stream.record_event()

        if timer.enabled:
            print(
                f"[{self.block_type}] Weight swap timing at {self.debug_block_count}:",
                flush=True,
            )
            for sec_name, sec_time in timer.sections.items():
                print(f"  {sec_name}: {sec_time * 1000:.2f}ms", flush=True)
            print(f"  TOTAL: {timer.total * 1000:.2f}ms", flush=True)

        return final_event

    # -- dispatch -----------------------------------------------------------

    def swap_weight_devices(
        self,
        block_to_cpu: nn.Module,
        block_to_cuda: nn.Module,
    ) -> Optional[torch.cuda.Event]:
        if self.cuda_available:
            return self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            _swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            return None

    # -- async submission / wait --------------------------------------------

    def _submit_move_blocks(
        self,
        blocks: list[nn.Module],
        block_idx_to_cpu: int,
        block_idx_to_cuda: int,
    ) -> None:
        device = self.device
        block_type = self.block_type
        debug = self.debug

        def move_blocks(
            bidx_to_cpu: int,
            block_to_cpu: nn.Module,
            bidx_to_cuda: int,
            block_to_cuda: nn.Module,
        ) -> tuple[int, int, Optional[torch.cuda.Event]]:
            # Set CUDA device in worker thread
            if device.type == "cuda":
                dev = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_device(dev)

            if debug:
                print(
                    f"[{block_type}] Move block {bidx_to_cpu} to CPU and "
                    f"block {bidx_to_cuda} to CUDA",
                    flush=True,
                )

            t0 = time.perf_counter()
            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)
            elapsed = time.perf_counter() - t0

            if debug:
                print(
                    f"[{block_type}] Move completed: {bidx_to_cpu}->CPU, "
                    f"{bidx_to_cuda}->CUDA in {elapsed:.3f}s",
                    flush=True,
                )

            return (bidx_to_cpu, bidx_to_cuda, sync_event)

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]
        future = self.thread_pool.submit(
            move_blocks,
            block_idx_to_cpu, block_to_cpu,
            block_idx_to_cuda, block_to_cuda,
        )
        self.futures[block_idx_to_cuda] = future

    def _wait_blocks_move(self, block_idx: int) -> None:
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}", flush=True)

        t0 = time.perf_counter()
        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda, (
            f"Expected block {block_idx} but got {bidx_to_cuda}"
        )

        if self.cuda_available and sync_event is not None:
            # GPU-side wait -- does NOT block the CPU
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            elapsed = time.perf_counter() - t0
            print(
                f"[{self.block_type}] Waited for block {block_idx}: {elapsed:.3f}s",
                flush=True,
            )


# ---------------------------------------------------------------------------
# BlockSwapScheduler
# ---------------------------------------------------------------------------

class BlockSwapScheduler(BlockSwapOffloader):
    """Pre-computed swap schedule for fixed forward/backward execution order.

    Forward:  blocks 0 -> N-1
    Backward: blocks N-1 -> 0

    GPU-resident blocks sit in [0, num_blocks - blocks_to_swap).
    CPU-offloaded blocks sit in [num_blocks - blocks_to_swap, num_blocks).

    During forward pass, GPU-resident blocks are swapped out to bring in
    CPU-offloaded blocks as they're needed.  Backward hooks reverse the process.
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            block_type, num_blocks, blocks_to_swap,
            device, use_pinned_memory, debug,
        )
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        if supports_backward:
            self.remove_handles: list = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def __del__(self) -> None:
        if getattr(self, "supports_backward", False) and hasattr(self, "remove_handles"):
            for handle in self.remove_handles:
                handle.remove()

    def set_forward_only(self, forward_only: bool) -> None:
        # Drain all pending futures first
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    # -- backward hooks (pre-computed at init) ------------------------------

    def create_backward_hook(
        self,
        blocks: list[nn.Module],
        block_index: int,
    ) -> Optional[callable]:
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # Pre-compute constant indices captured by the closure
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        block_type = self.block_type
        debug = self.debug

        def backward_hook(module, grad_input, grad_output):
            if self.forward_only:
                return None

            if debug:
                print(f"[{block_type}] Backward hook for block {block_index}", flush=True)

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)

            if waiting:
                self._wait_blocks_move(block_idx_to_wait)

            return None

        return backward_hook

    # -- public API ---------------------------------------------------------

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward", flush=True)

        # GPU-resident blocks: move everything to device
        for block in blocks[: self.num_blocks - self.blocks_to_swap]:
            block.to(self.device)
            _weighs_to_device(block, self.device)

        # CPU-offloaded blocks: buffers (norms, etc.) on GPU, Linear weights on CPU
        for block in blocks[self.num_blocks - self.blocks_to_swap :]:
            block.to(self.device)
            _weighs_to_device(block, torch.device("cpu"))

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(
        self,
        blocks: list[nn.Module],
        block_idx: int,
    ) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            # Backward enabled: only the first blocks_to_swap blocks need
            # swapping during forward -- the rest are already on GPU.
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        # Forward-only mode
        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < self.num_blocks // 2:
            # Strategy 1: few blocks offloaded, no wrap-around
            # Middle blocks are always on GPU -- no swap needed
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                block_idx_to_cuda = (
                    self.num_blocks - self.blocks_to_swap + block_idx
                ) % self.num_blocks
            else:
                # block_idx >= num_blocks - blocks_to_swap
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            # Strategy 2: many blocks offloaded, wrap-around
            block_idx_to_cuda = (
                self.num_blocks - self.blocks_to_swap + block_idx
            ) % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
