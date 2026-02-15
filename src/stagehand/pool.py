"""Fixed-size pinned host memory pool with backpressure.

The pool is pre-allocated at init and NEVER grows.  ``acquire()`` blocks
when all slabs are in use (backpressure) — it will never call
``torch.empty()`` or any other allocator at runtime.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from stagehand.errors import StagehandOOMError

__all__ = ["PinnedPool", "PinnedSlab"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PinnedSlab
# ---------------------------------------------------------------------------

@dataclass
class PinnedSlab:
    """Handle for a single pinned host buffer owned by a :class:`PinnedPool`."""

    slab_id: int
    buffer: torch.Tensor
    size_bytes: int
    pool_id: int


# ---------------------------------------------------------------------------
# PinnedPool
# ---------------------------------------------------------------------------

class PinnedPool:
    """Fixed-size pool of pinned host memory slabs.

    Pre-allocated at ``__init__``.  Never grows.  Ring-buffer recycling with
    backpressure via :pyclass:`threading.Condition`.

    Parameters
    ----------
    total_mb:
        Total pinned memory budget in MiB.
    slab_mb:
        Individual slab size in MiB.  Must evenly divide *total_mb*.
    alignment:
        Byte alignment hint (default 512 for DMA friendliness).
    """

    def __init__(
        self,
        total_mb: int = 8192,
        slab_mb: int = 256,
        alignment: int = 512,
    ) -> None:
        if total_mb <= 0:
            raise ValueError(f"total_mb must be positive, got {total_mb}")
        if slab_mb <= 0:
            raise ValueError(f"slab_mb must be positive, got {slab_mb}")
        if total_mb % slab_mb != 0:
            raise ValueError(
                f"total_mb ({total_mb}) must be evenly divisible by slab_mb ({slab_mb})"
            )

        self._pool_id: int = id(self)
        self._slab_mb: int = slab_mb
        # Round slab size up to the requested alignment for DMA friendliness.
        raw_slab_bytes = slab_mb * 1024 * 1024
        if alignment > 0:
            self._slab_bytes = ((raw_slab_bytes + alignment - 1) // alignment) * alignment
        else:
            self._slab_bytes = raw_slab_bytes
        self._alignment: int = alignment

        num_slabs = total_mb // slab_mb
        self._num_slabs: int = num_slabs

        # Determine if we can use pinned memory (requires CUDA).
        use_pin = torch.cuda.is_available()

        # Pre-allocate ALL slabs at init.
        self._all_slabs: list[PinnedSlab] = []
        self._free: deque[PinnedSlab] = deque()
        for i in range(num_slabs):
            buf = torch.empty(self._slab_bytes, dtype=torch.uint8, pin_memory=use_pin)
            slab = PinnedSlab(
                slab_id=i,
                buffer=buf,
                size_bytes=self._slab_bytes,
                pool_id=self._pool_id,
            )
            self._all_slabs.append(slab)
            self._free.append(slab)

        # Synchronisation primitive — Condition wraps its own Lock.
        self._cond = threading.Condition()

        # Stats (protected by self._cond).
        self._in_use: int = 0
        self._peak_in_use: int = 0
        self._wait_count: int = 0
        self._total_wait_ms: float = 0.0
        self._shutdown: bool = False

        log.info(
            "PinnedPool: %d slabs x %d MiB = %d MiB (pinned=%s)",
            num_slabs, slab_mb, total_mb, use_pin,
        )

    # ------------------------------------------------------------------
    # acquire
    # ------------------------------------------------------------------

    def acquire(self, size_bytes: int) -> PinnedSlab | list[PinnedSlab]:
        """Pop slab(s) from the free-list.  Blocks if none available.

        If *size_bytes* fits in a single slab, returns a single
        :class:`PinnedSlab`.  If it spans multiple slabs, returns a list.

        Raises
        ------
        StagehandOOMError
            If the request exceeds the total pool capacity.
        """
        slabs_needed = math.ceil(size_bytes / self._slab_bytes)
        if slabs_needed > self._num_slabs:
            raise StagehandOOMError(
                f"Requested {size_bytes} bytes ({slabs_needed} slabs) exceeds "
                f"pool capacity of {self._num_slabs} slabs "
                f"({self._num_slabs * self._slab_bytes} bytes)"
            )

        acquired: list[PinnedSlab] = []
        with self._cond:
            for _ in range(slabs_needed):
                # Block until a slab is available.
                warned = False
                wait_start: float | None = None
                while len(self._free) == 0:
                    if wait_start is None:
                        wait_start = time.monotonic()
                    # Wait with a timeout so we can emit a warning.
                    self._cond.wait(timeout=0.1)
                    if not warned and wait_start is not None:
                        elapsed_ms = (time.monotonic() - wait_start) * 1000.0
                        if elapsed_ms >= 100.0:
                            log.warning(
                                "PinnedPool.acquire: waited %.1f ms for a free slab",
                                elapsed_ms,
                            )
                            warned = True

                # Record wait stats if we blocked.
                if wait_start is not None:
                    elapsed_ms = (time.monotonic() - wait_start) * 1000.0
                    self._wait_count += 1
                    self._total_wait_ms += elapsed_ms

                slab = self._free.popleft()
                acquired.append(slab)
                self._in_use += 1
                if self._in_use > self._peak_in_use:
                    self._peak_in_use = self._in_use

        if slabs_needed == 1:
            return acquired[0]
        return acquired

    # ------------------------------------------------------------------
    # release
    # ------------------------------------------------------------------

    def release(self, slab: PinnedSlab | list[PinnedSlab]) -> None:
        """Return slab(s) to the free-list and wake waiting threads."""
        if isinstance(slab, PinnedSlab):
            slabs = [slab]
        else:
            slabs = list(slab)

        with self._cond:
            for s in slabs:
                assert s.pool_id == self._pool_id, (
                    f"Slab pool_id {s.pool_id} does not match pool {self._pool_id}"
                )
                self._free.append(s)
                self._in_use -= 1
            # Wake ALL waiters so multi-slab acquires can make progress.
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int | float]:
        """Return current pool statistics."""
        with self._cond:
            return {
                "total": self._num_slabs,
                "free": len(self._free),
                "in_use": self._in_use,
                "peak_in_use": self._peak_in_use,
                "wait_count": self._wait_count,
                "total_wait_ms": self._total_wait_ms,
            }

    # ------------------------------------------------------------------
    # shutdown / cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all slab buffers.  Pool is unusable after this."""
        with self._cond:
            self._shutdown = True
            self._free.clear()
            for s in self._all_slabs:
                s.buffer = None  # type: ignore[assignment]
            self._all_slabs.clear()
            self._cond.notify_all()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def slab_bytes(self) -> int:
        """Size of a single slab in bytes."""
        return self._slab_bytes

    @property
    def num_slabs(self) -> int:
        """Total number of slabs in the pool."""
        return self._num_slabs
