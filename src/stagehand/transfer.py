"""Async H2D/D2H transfer engine with backpressure.

Manages asynchronous copies between pinned host memory (PinnedPool slabs)
and GPU tensors using a dedicated CUDA stream and events.  Bounded by
``max_inflight`` to prevent unbounded queue growth.

Falls back to synchronous copies when CUDA is unavailable (CPU testing).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from stagehand.pool import PinnedPool, PinnedSlab

__all__ = ["TransferHandle", "AsyncTransferEngine"]

log = logging.getLogger(__name__)
_STALL_LOGS_ENABLED = str(os.getenv("SERENITY_STAGEHAND_STALL_LOGS", "0")).strip().lower() in {"1", "true", "yes", "on"}


# ── handle ──────────────────────────────────────────────────────────────


@dataclass
class TransferHandle:
    """Opaque handle returned by submit_h2d / submit_d2h."""

    handle_id: int
    block_id: str
    direction: str  # "h2d" or "d2h"
    event: torch.cuda.Event | None  # None when CUDA unavailable
    submitted_at: float  # time.monotonic()
    size_bytes: int
    completed: bool = False


# ── helpers ─────────────────────────────────────────────────────────────


def _reinterpret_slab(
    buffer: torch.Tensor,
    dtype: torch.dtype,
    shape: torch.Size,
) -> torch.Tensor:
    """View the first *numel * element_size* bytes of a uint8 buffer as *dtype*/*shape*.

    This is the standard pattern: slab buffers are stored as raw ``uint8``
    bytes, and we reinterpret them as the target dtype for the copy.
    """
    numel = 1
    for s in shape:
        numel *= s
    element_size = torch.empty(0, dtype=dtype).element_size()
    nbytes = numel * element_size
    return buffer[:nbytes].view(dtype).reshape(shape)


# ── engine ──────────────────────────────────────────────────────────────


class AsyncTransferEngine:
    """Manages async H2D/D2H copies on a dedicated CUDA stream.

    Parameters
    ----------
    pool:
        The :class:`PinnedPool` that owns host-side slabs.
    max_inflight:
        Maximum concurrent transfers before backpressure kicks in.
    copy_stream:
        Optional pre-created CUDA stream.  If *None* and CUDA is available,
        a high-priority stream is created automatically.
    """

    def __init__(
        self,
        pool: PinnedPool,
        max_inflight: int = 2,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._pool = pool
        self._max_inflight = max_inflight
        self._has_cuda = torch.cuda.is_available()

        if self._has_cuda:
            self._copy_stream = copy_stream or torch.cuda.Stream(
                priority=-1,  # high priority
            )
        else:
            self._copy_stream = None

        # Inflight tracking — protected by _lock / _cond.
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._inflight: list[TransferHandle] = []
        self._next_handle_id: int = 0

    # ── public API ────────────────────────────────────────────────────

    def submit_h2d(
        self,
        block_id: str,
        host_slab: PinnedSlab,
        gpu_dest: torch.Tensor,
    ) -> TransferHandle:
        """Copy *host_slab* buffer to *gpu_dest* asynchronously.

        Blocks if ``inflight_count >= max_inflight`` (backpressure).
        """
        size_bytes = gpu_dest.numel() * gpu_dest.element_size()
        self._wait_for_slot()

        handle = self._make_handle(block_id, "h2d", size_bytes)

        # Reinterpret the uint8 slab buffer as the destination dtype.
        src = _reinterpret_slab(host_slab.buffer, gpu_dest.dtype, gpu_dest.shape)

        if self._has_cuda:
            # Ensure copy_stream waits for any pending work on the default
            # stream that may have produced gpu_dest (e.g. torch.zeros).
            default_event = torch.cuda.Event()
            default_event.record(torch.cuda.current_stream())
            self._copy_stream.wait_event(default_event)
            with torch.cuda.stream(self._copy_stream):
                gpu_dest.copy_(src, non_blocking=True)
                handle.event = torch.cuda.Event()
                handle.event.record(self._copy_stream)
        else:
            gpu_dest.copy_(src)
            handle.completed = True

        with self._lock:
            self._inflight.append(handle)

        return handle

    def submit_d2h(
        self,
        block_id: str,
        gpu_src: torch.Tensor,
        host_slab: PinnedSlab,
    ) -> TransferHandle:
        """Copy *gpu_src* to *host_slab* buffer asynchronously."""
        size_bytes = gpu_src.numel() * gpu_src.element_size()
        self._wait_for_slot()

        handle = self._make_handle(block_id, "d2h", size_bytes)

        # Reinterpret the uint8 slab buffer as the source dtype.
        dst = _reinterpret_slab(host_slab.buffer, gpu_src.dtype, gpu_src.shape)

        if self._has_cuda:
            # Ensure copy_stream waits for any pending work on the default
            # stream that may have produced gpu_src.
            default_event = torch.cuda.Event()
            default_event.record(torch.cuda.current_stream())
            self._copy_stream.wait_event(default_event)
            with torch.cuda.stream(self._copy_stream):
                dst.copy_(gpu_src, non_blocking=True)
                handle.event = torch.cuda.Event()
                handle.event.record(self._copy_stream)
        else:
            dst.copy_(gpu_src)
            handle.completed = True

        with self._lock:
            self._inflight.append(handle)

        return handle

    def poll(self, handle: TransferHandle) -> bool:
        """Non-blocking check whether *handle* has completed."""
        if handle.completed:
            return True
        if handle.event is None:
            # No event: either CPU mode (already done) or manually injected.
            if not self._has_cuda:
                return True
            return handle.completed
        done = handle.event.query()
        if done:
            self._complete_handle(handle)
        return done

    def wait(self, handle: TransferHandle) -> None:
        """Block until *handle* completes.  Logs a stall if wait > 1 ms."""
        if handle.completed:
            return
        if handle.event is None:
            # No CUDA event — either CPU mode or handle has no tracking.
            self._complete_handle(handle)
            return

        t0 = time.monotonic()
        handle.event.synchronize()
        # Ensure the default stream also waits for the copy to complete,
        # so subsequent reads on the default stream see the transferred data.
        torch.cuda.current_stream().wait_event(handle.event)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if _STALL_LOGS_ENABLED and elapsed_ms > 1.0:
            log.warning(
                "Transfer stall: block=%s dir=%s wait=%.2f ms",
                handle.block_id,
                handle.direction,
                elapsed_ms,
            )

        self._complete_handle(handle)

    def inflight_count(self) -> int:
        """Number of transfers currently in flight."""
        with self._lock:
            return len(self._inflight)

    def reap(self) -> int:
        """Reap completed transfers.  Returns number reaped."""
        with self._lock:
            before = len(self._inflight)
            self._reap_completed_locked()
            return before - len(self._inflight)

    def drain(self) -> None:
        """Wait for all inflight transfers to complete."""
        while True:
            with self._lock:
                remaining = list(self._inflight)
            if not remaining:
                break
            for h in remaining:
                self.wait(h)

    # ── internal helpers ──────────────────────────────────────────────

    def _make_handle(
        self, block_id: str, direction: str, size_bytes: int
    ) -> TransferHandle:
        with self._lock:
            hid = self._next_handle_id
            self._next_handle_id += 1
        return TransferHandle(
            handle_id=hid,
            block_id=block_id,
            direction=direction,
            event=None,
            submitted_at=time.monotonic(),
            size_bytes=size_bytes,
        )

    def _wait_for_slot(self) -> None:
        """Block until an inflight slot is available (backpressure)."""
        t0 = time.monotonic()
        warned = False
        with self._cond:
            while len(self._inflight) >= self._max_inflight:
                # Try reaping completed handles first.
                self._reap_completed_locked()
                if len(self._inflight) < self._max_inflight:
                    break
                self._cond.wait(timeout=0.001)
                if not warned:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    if elapsed_ms > 1.0:
                        log.warning(
                            "AsyncTransferEngine: backpressure wait %.2f ms",
                            elapsed_ms,
                        )
                        warned = True

    def _complete_handle(self, handle: TransferHandle) -> None:
        """Mark handle done and free its inflight slot."""
        if handle.completed:
            return
        handle.completed = True
        self._remove_from_inflight(handle)

    def _remove_from_inflight(self, handle: TransferHandle) -> None:
        with self._cond:
            try:
                self._inflight.remove(handle)
            except ValueError:
                pass  # already removed
            self._cond.notify_all()

    def _reap_completed(self) -> None:
        """Poll all inflight handles and complete any finished ones."""
        with self._lock:
            self._reap_completed_locked()

    def _reap_completed_locked(self) -> None:
        """Must be called with _lock held."""
        to_remove: list[TransferHandle] = []
        for h in self._inflight:
            if h.completed:
                to_remove.append(h)
            elif h.event is not None and h.event.query():
                h.completed = True
                to_remove.append(h)
            elif h.event is None and not self._has_cuda:
                # CPU-only fallback: no event means already done.
                h.completed = True
                to_remove.append(h)
        for h in to_remove:
            self._inflight.remove(h)
        if to_remove:
            self._cond.notify_all()
