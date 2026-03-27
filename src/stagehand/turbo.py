"""Stagehand Turbo — Rust-accelerated transfer engine.

Pre-allocates torch GPU tensors and pinned CPU tensors. The Rust engine
does cudaMemcpyAsync between their data_ptrs on dedicated CUDA streams.
Python just swaps param.data pointers — zero-copy on the Python side.
"""
from __future__ import annotations

import ctypes
import logging
import re

import torch
from torch import nn

log = logging.getLogger(__name__)

try:
    import stagehand_turbo
    _TURBO_AVAILABLE = True
except ImportError:
    _TURBO_AVAILABLE = False


def is_available() -> bool:
    return _TURBO_AVAILABLE


class TurboConductor:
    """Block-swap conductor backed by the Rust transfer engine.

    GPU memory is allocated as a small window of torch tensors (3 slots).
    The Rust engine rotates blocks through these slots via cudaMemcpyAsync
    on dedicated streams, releasing the GIL during transfers.
    """

    def __init__(
        self,
        model: nn.Module,
        block_pattern: str = r".*\.block\.\d+$",
        dtype: torch.dtype = torch.bfloat16,
        device: int = 0,
        vram_budget_gb: float = 18.0,
    ) -> None:
        if not _TURBO_AVAILABLE:
            raise RuntimeError("stagehand_turbo not installed")

        self._dtype = dtype
        self._device_idx = device
        self._device = torch.device(f"cuda:{device}")

        # Discover blocks
        pat = re.compile(block_pattern)
        self._block_modules: list[nn.Module] = []
        self._block_names: list[str] = []
        for name, mod in model.named_modules():
            if pat.match(name):
                self._block_names.append(name)
                self._block_modules.append(mod)
        if not self._block_modules:
            raise ValueError(f"No blocks matched {block_pattern!r}")

        self._engine = None
        self._prepared = False

        # Per-block param metadata
        self._param_meta: list[list[tuple[str, tuple[int, ...], int]]] = []  # (name, shape, nbytes)

        # Per-slot: pre-allocated torch GPU tensors
        self._slot_tensors: list[list[torch.Tensor]] = []  # [slot_idx][param_idx]

        # Per-block: pinned CPU tensors (always allocated for all blocks)
        self._cpu_tensors: list[list[torch.Tensor]] = []  # [block_idx][param_idx]

        log.info("TurboConductor: %d blocks", len(self._block_modules))

    @property
    def num_blocks(self) -> int:
        return len(self._block_modules)

    def prepare(self) -> None:
        if self._prepared:
            return

        engine = stagehand_turbo.TransferEngine()

        # 1. Register blocks
        for mod in self._block_modules:
            sizes = []
            meta = []
            for name, p in mod.named_parameters():
                nbytes = p.data.nelement() * self._dtype.itemsize
                sizes.append(nbytes)
                meta.append((name, tuple(p.shape), nbytes))
            engine.register_block(sizes)
            self._param_meta.append(meta)

        # 2. Initialize Rust engine with window=3
        window_size = 3
        engine.initialize(self._device_idx, window_size)

        # 3. Stage weights to Rust's pinned pool
        for block_idx, mod in enumerate(self._block_modules):
            src_ptrs = []
            src_tensors = []  # prevent GC before stage_block copies data
            for name, p in mod.named_parameters():
                d = p.data.detach().cpu().to(self._dtype).contiguous()
                src_ptrs.append(d.data_ptr())
                src_tensors.append(d)
            engine.stage_block(block_idx, src_ptrs)
            del src_tensors  # safe to release after memcpy completes

        # 4. Build CPU pinned tensor views
        for block_idx in range(len(self._block_modules)):
            cpu_views = []
            for param_idx, (pname, shape, nbytes) in enumerate(self._param_meta[block_idx]):
                cpu_ptr = engine.cpu_ptr(block_idx, param_idx)
                buf = (ctypes.c_char * nbytes).from_address(cpu_ptr)
                cpu_t = torch.frombuffer(buf, dtype=self._dtype).reshape(shape)
                cpu_views.append(cpu_t)
            self._cpu_tensors.append(cpu_views)

        # 5. Pre-allocate torch GPU tensors for each slot.
        # Each slot has one tensor per param, sized to the largest block's param.
        max_params = max(len(m) for m in self._param_meta)
        for slot_idx in range(window_size):
            slot_views = []
            for param_idx in range(max_params):
                # Find max shape for this param position across all blocks
                max_numel = 0
                for block_meta in self._param_meta:
                    if param_idx < len(block_meta):
                        _, shape, _ = block_meta[param_idx]
                        numel = 1
                        for s in shape:
                            numel *= s
                        max_numel = max(max_numel, numel)
                if max_numel > 0:
                    gpu_t = torch.empty(max_numel, dtype=self._dtype, device=self._device)
                else:
                    gpu_t = torch.empty(0, dtype=self._dtype, device=self._device)
                slot_views.append(gpu_t)

            # Tell Rust where these GPU tensors live
            ptrs = [t.data_ptr() for t in slot_views]
            engine.set_slot_gpu_ptrs(slot_idx, ptrs)
            self._slot_tensors.append(slot_views)

        # 6. Point all params at CPU pinned memory
        for block_idx, mod in enumerate(self._block_modules):
            for param_idx, (name, p) in enumerate(mod.named_parameters()):
                p.data = self._cpu_tensors[block_idx][param_idx]

        self._engine = engine
        self._prepared = True
        log.info("TurboConductor: ready, %d slots, %d blocks", window_size, len(self._block_modules))

    def before_block(self, block_idx: int, direction: str = "forward") -> None:
        """Ensure block on GPU. Releases GIL during CUDA sync."""
        assert self._engine is not None
        self._engine.step(block_idx, direction)

        # Get which slot this block landed in
        slot_idx = self._engine.block_slot(block_idx)
        if slot_idx is None:
            raise RuntimeError(f"Block {block_idx} has no GPU slot after step()")

        # Swap param.data to point at the slot's GPU tensor (reshaped)
        mod = self._block_modules[block_idx]
        for param_idx, (name, p) in enumerate(mod.named_parameters()):
            shape = self._param_meta[block_idx][param_idx][1]
            numel = 1
            for s in shape:
                numel *= s
            # View the slot tensor with the right shape
            p.data = self._slot_tensors[slot_idx][param_idx][:numel].reshape(shape)

    def after_block(self, block_idx: int) -> None:
        """Point params back to CPU pinned."""
        mod = self._block_modules[block_idx]
        for param_idx, (name, p) in enumerate(mod.named_parameters()):
            p.data = self._cpu_tensors[block_idx][param_idx]

    def evict_all(self) -> None:
        if self._engine is None:
            return
        for idx in range(len(self._block_modules)):
            loc = self._engine.block_location(idx)
            if loc in ("gpu", "h2d"):
                self._engine.evict_sync(idx)
        for block_idx, mod in enumerate(self._block_modules):
            for param_idx, (name, p) in enumerate(mod.named_parameters()):
                p.data = self._cpu_tensors[block_idx][param_idx]

    def shutdown(self) -> None:
        # Release GPU tensors BEFORE engine (prevents use-after-free)
        self._slot_tensors.clear()
        self._cpu_tensors.clear()
        if self._engine is not None:
            self._engine.drain()
            self._engine = None
