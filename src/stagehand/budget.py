"""VRAM budget manager with watermark-based eviction control.

Wraps ``torch.cuda.memory_allocated()`` / ``memory_reserved()`` and
exposes simple predicates the scheduler uses to decide when to evict
and when prefetching is safe.

Default watermarks are tuned for a 3090 (24 GB VRAM) leaving 2-2.5 GB
headroom for activations, optimizer states, and the PyTorch caching
allocator.  First target model: WAN 2.2.
"""
from __future__ import annotations

import torch

__all__ = ["BudgetManager"]


class BudgetManager:
    """Tracks VRAM usage and enforces watermark thresholds.

    Parameters
    ----------
    high_watermark_mb:
        Trigger eviction above this level.
    low_watermark_mb:
        Stop evicting once usage drops below this level.

    Raises
    ------
    ValueError
        If *high_watermark_mb* <= *low_watermark_mb*.
    """

    def __init__(
        self,
        high_watermark_mb: int = 22000,
        low_watermark_mb: int = 18000,
    ) -> None:
        if high_watermark_mb <= low_watermark_mb:
            raise ValueError(
                f"high_watermark_mb ({high_watermark_mb}) must be greater "
                f"than low_watermark_mb ({low_watermark_mb})"
            )
        self._high_watermark_mb = high_watermark_mb
        self._low_watermark_mb = low_watermark_mb
        self._reserved_bytes: int = 0

    # ── raw queries ──────────────────────────────────────────────────

    def vram_used_mb(self) -> float:
        """Current VRAM allocated by PyTorch (MB).  Returns 0.0 if no CUDA."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)

    def vram_reserved_mb(self) -> float:
        """Current VRAM reserved by the caching allocator (MB).  Returns 0.0 if no CUDA."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved() / (1024 * 1024)

    # ── predicates ───────────────────────────────────────────────────

    def above_high_watermark(self) -> bool:
        """True if VRAM usage exceeds the high watermark."""
        return self.vram_used_mb() >= self._high_watermark_mb

    def below_low_watermark(self) -> bool:
        """True if VRAM usage is below the low watermark."""
        return self.vram_used_mb() < self._low_watermark_mb

    def headroom_mb(self) -> float:
        """Remaining MB before the high watermark is hit."""
        return self._high_watermark_mb - self.vram_used_mb()

    def should_evict(self) -> bool:
        """Alias for :meth:`above_high_watermark`."""
        return self.above_high_watermark()

    def can_prefetch(self) -> bool:
        """True if usage is below the high watermark (normal ops or opportunistic)."""
        return not self.above_high_watermark()

    # ── reservation ────────────────────────────────────────────────

    def reserve_bytes(self, amount: int, label: str = "") -> None:
        """Subtract *amount* from effective budget available for guest/normal models.

        Does not allocate VRAM — it constrains future guest allocation decisions.
        """
        self._reserved_bytes += amount

    def release_reserved_bytes(self, amount: int) -> None:
        """Release previously reserved bytes back to the guest pool."""
        self._reserved_bytes = max(0, self._reserved_bytes - amount)

    @property
    def reserved_bytes(self) -> int:
        """Total bytes currently reserved for protected resident models."""
        return self._reserved_bytes

    def can_guest_allocate(self, requested_bytes: int, safety_margin_mb: int = 500) -> bool:
        """Check if a guest model can load without displacing protected residents.

        Evaluates against headroom AFTER subtracting reserved bytes and a
        safety margin from the total budget.
        """
        budget_bytes = self._high_watermark_mb * 1024 * 1024
        used_bytes = int(self.vram_used_mb() * 1024 * 1024)
        safety_bytes = safety_margin_mb * 1024 * 1024
        effective_available = budget_bytes - used_bytes - self._reserved_bytes - safety_bytes
        return requested_bytes <= effective_available

    # ── introspection ────────────────────────────────────────────────

    @property
    def high_watermark_mb(self) -> int:
        return self._high_watermark_mb

    @property
    def low_watermark_mb(self) -> int:
        return self._low_watermark_mb

    def __repr__(self) -> str:
        reserved_mb = self._reserved_bytes / (1024 * 1024)
        return (
            f"BudgetManager(high={self._high_watermark_mb}MB, "
            f"low={self._low_watermark_mb}MB, "
            f"used={self.vram_used_mb():.0f}MB, "
            f"reserved={reserved_mb:.0f}MB)"
        )
