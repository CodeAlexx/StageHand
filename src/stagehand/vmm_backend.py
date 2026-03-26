"""VMM backend for Stagehand inference.

Provides zero-overhead weight residency via CUDA Virtual Memory Management.
Falls back gracefully if stagehand_vmm is not available or GPU is not Ampere+.

This module is optional — Stagehand works without it (block-swap fallback).
"""
from __future__ import annotations

import ctypes
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_vmm = None
_vmm_available: bool | None = None


def is_available() -> bool:
    """Check if VMM backend is usable on this system."""
    global _vmm, _vmm_available
    if _vmm_available is not None:
        return _vmm_available

    try:
        import stagehand_vmm
        _vmm = stagehand_vmm

        if not torch.cuda.is_available():
            _vmm_available = False
            return False

        cc = torch.cuda.get_device_capability()
        if cc[0] < 8:
            logger.info(
                "VMM requires Ampere+ (SM 8.0+), found SM %d.%d. "
                "Using block-swap fallback.", cc[0], cc[1],
            )
            _vmm_available = False
            return False

        _vmm_available = True
        logger.info("VMM backend available.")
        return True

    except ImportError:
        logger.info("stagehand_vmm not installed. Using block-swap fallback.")
        _vmm_available = False
        return False
    except Exception as e:
        logger.warning("VMM backend init failed: %s. Using block-swap fallback.", e)
        _vmm_available = False
        return False


# ── dtype helpers ─────────────────────────────────────────────────────────

_SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "U8": torch.uint8,
}


def _safetensors_dtype_to_torch(dtype_str: str) -> torch.dtype:
    return _SAFETENSORS_DTYPE_MAP.get(dtype_str, torch.float32)


def _tensor_from_ptr(ptr: int, size_bytes: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a CPU tensor that views mmap'd memory at ptr. Zero-copy.

    The returned tensor does NOT own the memory — the MmapFile must stay
    alive for the tensor's lifetime. Use only for transient copy operations.
    """
    c_buf = (ctypes.c_uint8 * size_bytes).from_address(ptr)
    return torch.frombuffer(c_buf, dtype=torch.uint8, count=size_bytes)


# ── VmmModelHandle ────────────────────────────────────────────────────────

class VmmModelHandle:
    """Manages VMM slab and regions for one model's transformer blocks."""

    def __init__(
        self,
        allocator: object,
        model_id: str,
        block_sizes: list[int],
        dtype: str,
    ) -> None:
        self.allocator = allocator
        self.model_id = model_id
        self.dtype = dtype
        self.block_count = len(block_sizes)
        self.block_sizes = block_sizes

        total_size = sum(block_sizes)
        self.slab_id = allocator.create_slab(total_size)

        self.regions: list[int] = []
        self.block_shapes: list[list[int] | None] = [None] * len(block_sizes)
        offset = 0
        for size in block_sizes:
            rid = allocator.define_region(self.slab_id, offset, size)
            self.regions.append(rid)
            offset += size

        self._active_handles: dict[int, object] = {}
        self._populated: list[bool] = [False] * len(block_sizes)
        self._current_priority: int = 0
        self._destroyed = False

        # Weight sources — mmap preferred, committed RAM as fallback
        self._mmap_file: object | None = None
        self._block_tensor_names: list[list[str]] = []
        self._block_byte_ranges: list[tuple[int, int]] = []
        self._ram_weight_refs: list[torch.Tensor | None] = [None] * len(block_sizes)

        logger.debug(
            "VMM: slab for %s: %d regions, %.2fGB",
            model_id, len(block_sizes), total_size / 1e9,
        )

    def register_block_shape(self, block_idx: int, shape: list[int]) -> None:
        """Register the flat tensor shape for a block."""
        self.block_shapes[block_idx] = shape

    # ── weight source setup ───────────────────────────────────────────

    def set_mmap_source(
        self,
        safetensors_path: str,
        block_tensor_map: list[list[str]],
    ) -> None:
        """Set an mmap'd safetensors file as the weight source.

        Weights are read directly from the file via the OS page cache —
        no committed RAM allocation. The OS manages pages transparently:
        pages are brought in on demand and reclaimed freely under pressure.

        Args:
            safetensors_path: path to the .safetensors file
            block_tensor_map: for each block, the tensor names in that block
        """
        try:
            from serenity_safetensors import MmapFile
            if MmapFile is None:
                raise ImportError("MmapFile not available (non-Linux?)")

            self._mmap_file = MmapFile(safetensors_path)
            self._block_tensor_names = block_tensor_map

            # Compute byte ranges for dual-layer prefetch
            self._block_byte_ranges = []
            for block_names in block_tensor_map:
                offsets = []
                for name in block_names:
                    try:
                        offset, size, _, _ = self._mmap_file.tensor_info(name)
                        offsets.append((offset, size))
                    except KeyError:
                        pass
                if offsets:
                    start = min(o[0] for o in offsets)
                    end = max(o[0] + o[1] for o in offsets)
                    self._block_byte_ranges.append((start, end - start))
                else:
                    self._block_byte_ranges.append((0, 0))

            logger.info(
                "VMM: mmap source set for %s: %s (%.2fGB)",
                self.model_id, safetensors_path,
                self._mmap_file.data_size() / 1e9,
            )

        except (ImportError, Exception) as e:
            logger.warning(
                "VMM: mmap source failed for %s: %s. "
                "Falling back to committed RAM weights.", self.model_id, e,
            )
            self._mmap_file = None

    def set_ram_weights(self, block_idx: int, weights: torch.Tensor) -> None:
        """Store committed RAM weight reference (fallback when mmap unavailable)."""
        self._ram_weight_refs[block_idx] = weights

    # ── block access ──────────────────────────────────────────────────

    def get_block_tensor(
        self, block_idx: int, stream: int,
    ) -> tuple[torch.Tensor | None, bool]:
        """Get a VMM-backed tensor for a block.

        Returns (tensor, True) on success, (None, False) if watermarked.
        """
        if self.block_shapes[block_idx] is None:
            raise RuntimeError(
                f"Block {block_idx} shape not registered. "
                f"Call register_block_shape first."
            )

        handle = None
        try:
            handle = self.allocator.ensure_resident(
                self.slab_id, self.regions[block_idx], stream,
            )
            tensor = handle.as_tensor(self.dtype, self.block_shapes[block_idx])

            if not self._populated[block_idx]:
                self._populate_block(block_idx, tensor)
                self._populated[block_idx] = True

            self._active_handles[block_idx] = handle
            return tensor, True

        except MemoryError:
            if handle is not None:
                handle.release()
            return None, False
        except Exception:
            if handle is not None:
                handle.release()
            raise

    def _populate_block(self, block_idx: int, vmm_tensor: torch.Tensor) -> None:
        """Copy weight data into VMM tensor from best available source."""
        if self._mmap_file is not None:
            self._populate_from_mmap(block_idx, vmm_tensor)
        elif self._ram_weight_refs[block_idx] is not None:
            vmm_tensor.copy_(self._ram_weight_refs[block_idx], non_blocking=True)
        else:
            raise RuntimeError(f"No weight source for block {block_idx}")

    def _populate_from_mmap(self, block_idx: int, vmm_tensor: torch.Tensor) -> None:
        """Copy block weights from mmap'd file to VMM tensor.

        The mmap source is uncommitted — the OS reads pages from disk on
        demand. After the async copy to VRAM completes, the OS can reclaim
        those RAM pages freely (they're clean file-backed pages).

        SAFETY: The non_blocking copy issues async DMA from mmap'd memory.
        self._mmap_file MUST stay alive until the copy completes. Do NOT
        call destroy() concurrently — synchronize on the CUDA stream first.
        """
        tensor_names = self._block_tensor_names[block_idx]

        # Build a flat uint8 CPU view from mmap'd pointers, matching the
        # byte-level layout that _restore_params_from_tensor expects.
        parts: list[torch.Tensor] = []
        for name in tensor_names:
            ptr = self._mmap_file.tensor_ptr(name)
            _, size, _, _ = self._mmap_file.tensor_info(name)
            # Create a transient CPU tensor viewing the mmap'd memory (zero-copy)
            cpu_view = _tensor_from_ptr(ptr, size, torch.uint8)
            parts.append(cpu_view)

        # Copy to VMM VRAM. View vmm_tensor as uint8 for byte-level copy.
        vmm_uint8 = vmm_tensor.view(torch.uint8)
        if len(parts) == 1:
            vmm_uint8.copy_(parts[0], non_blocking=True)
        else:
            combined = torch.cat(parts)
            vmm_uint8.copy_(combined, non_blocking=True)

    # ── prefetch ──────────────────────────────────────────────────────

    def prefetch_block(self, block_idx: int) -> None:
        """Prefetch a block — both VRAM mapping and RAM page cache."""
        if not (0 <= block_idx < self.block_count):
            return

        # VMM layer: map physical VRAM (async prefetch worker)
        try:
            self.allocator.prefetch(self.slab_id, self.regions[block_idx])
        except Exception:
            pass

        # RAM layer: warm OS page cache (madvise MADV_WILLNEED)
        if (
            self._mmap_file is not None
            and block_idx < len(self._block_byte_ranges)
            and not self._populated[block_idx]
        ):
            offset, size = self._block_byte_ranges[block_idx]
            if size > 0:
                self._mmap_file.prefetch_range(offset, size)

    # ── lifecycle ─────────────────────────────────────────────────────

    def release_block(self, block_idx: int) -> None:
        """Release the handle for a block after the layer has executed.

        The caller must detach module params from the VMM tensor BEFORE calling
        this, otherwise the DLPack tensor keeps the region pinned (refcount > 0).
        """
        handle = self._active_handles.pop(block_idx, None)
        if handle is not None:
            handle.release()

    def set_priority(self, priority: int) -> None:
        """Set eviction priority. Higher = evict last."""
        old = self._current_priority
        self._current_priority = priority
        self.allocator.set_priority(self.slab_id, priority)
        if priority < old:
            self._populated = [False] * self.block_count

    def invalidate_weights(self) -> None:
        """Mark all blocks as needing re-population (e.g. after LoRA merge)."""
        self._populated = [False] * self.block_count

    def release_all(self) -> None:
        """Release all active handles."""
        for idx in list(self._active_handles):
            self.release_block(idx)

    def release_mmap(self) -> None:
        """Release mmap pages to the OS (MADV_DONTNEED).

        Call when this model is deprioritized and won't be used soon.
        The data isn't lost — re-access reads from disk transparently.
        """
        if self._mmap_file is not None:
            self._mmap_file.release_to_os()

    def destroy(self) -> None:
        """Destroy the slab and release all resources."""
        if self._destroyed:
            return
        self._destroyed = True
        self.release_all()
        self._ram_weight_refs = [None] * self.block_count
        # Sync CUDA before dropping mmap — async copies (non_blocking=True)
        # in _populate_from_mmap may still be reading from mmap'd pages via
        # DMA. Dropping the mmap while DMA is in-flight is use-after-free.
        if self._mmap_file is not None and torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        self._mmap_file = None
        try:
            self.allocator.destroy_slab(self.slab_id)
        except Exception as e:
            logger.warning("VMM: destroy_slab failed for %s: %s", self.model_id, e)

    def __del__(self) -> None:
        if self._destroyed:
            return
        if self._active_handles:
            logger.warning(
                "VMM: __del__ called on %s with %d active handles — "
                "leaking slab to avoid use-after-free. Call destroy() explicitly.",
                self.model_id, len(self._active_handles),
            )
            return
        self.destroy()


# ── VmmManager ────────────────────────────────────────────────────────────

class VmmManager:
    """Manages the VMM allocator and per-model handles."""

    def __init__(
        self, device: int = 0, ceiling_mb: Optional[int] = None,
    ) -> None:
        if not is_available():
            raise RuntimeError("VMM backend not available")

        self.allocator = _vmm.SlabAllocator(device=device, ceiling_mb=ceiling_mb)
        self.models: dict[str, VmmModelHandle] = {}
        self._priority_counter = 0

        stats = self.allocator.stats()
        logger.info(
            "VMM: allocator ready. ceiling=%.2fGB, granularity=%.1fMB",
            stats["vram_ceiling"] / 1e9,
            stats["granularity"] / 1e6,
        )

    def register_model(
        self, model_id: str, block_sizes: list[int], dtype: str,
    ) -> VmmModelHandle:
        if model_id in self.models:
            logger.warning("VMM: %s already registered, replacing", model_id)
            self.models[model_id].destroy()

        handle = VmmModelHandle(self.allocator, model_id, block_sizes, dtype)
        self.models[model_id] = handle
        return handle

    def activate_model(self, model_id: str) -> None:
        """Promote a model to highest priority. Release mmap pages of other models."""
        if model_id not in self.models:
            return
        self._priority_counter += 1
        self.models[model_id].set_priority(self._priority_counter)

        # Release mmap pages for deprioritized models
        for mid, handle in self.models.items():
            if mid != model_id:
                handle.release_mmap()

    def unregister_model(self, model_id: str) -> None:
        handle = self.models.pop(model_id, None)
        if handle is not None:
            handle.destroy()

    def set_ceiling(self, ceiling_bytes: int) -> None:
        self.allocator.set_vram_ceiling(ceiling_bytes)

    def stats(self) -> dict:
        """Get allocator stats augmented with mmap info.

        The mmap_data_bytes field reports the total mmap'd data segment size
        across all models. This memory appears in RSS (top/htop) but is
        reclaimable — the OS manages it as page cache. Users should NOT add
        mmap_data_bytes to committed VRAM/RAM usage.
        """
        s = self.allocator.stats()
        # Track mmap'd file-backed memory separately from committed RAM.
        # These pages inflate RSS in top/htop but are freely reclaimable
        # by the OS (they're clean, file-backed page cache).
        mmap_bytes = 0
        for handle in self.models.values():
            if handle._mmap_file is not None:
                try:
                    mmap_bytes += handle._mmap_file.data_size()
                except Exception:
                    pass
        s["mmap_data_bytes"] = mmap_bytes
        s["mmap_data_gb"] = round(mmap_bytes / 1e9, 2)
        return s

    def shutdown(self) -> None:
        for mid in list(self.models):
            self.unregister_model(mid)
