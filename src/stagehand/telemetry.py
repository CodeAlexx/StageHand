"""Per-step and rolling-window telemetry for the stagehand runtime.

Tracks H2D/D2H bytes, stalls, evictions, prefetch hit rates, VRAM usage,
and numeric guard events.  Optionally writes a JSONL file for post-analysis
and prints a compact summary line every *interval_steps*.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import IO

__all__ = ["StepMetrics", "StagehandTelemetry"]

log = logging.getLogger(__name__)


# ── per-step metrics ────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    """Metrics collected for a single training step."""

    step: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    copy_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    stall_time_ms: float = 0.0
    stall_count: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    vram_used_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    host_pool_free: int = 0
    host_pool_in_use: int = 0
    dtype_promotions: int = 0
    nan_count: int = 0
    inf_count: int = 0


# ── telemetry engine ────────────────────────────────────────────────────


class StagehandTelemetry:
    """Collects, aggregates, and reports stagehand runtime metrics.

    Parameters
    ----------
    enabled:
        Master switch.  When *False*, all record_* calls are no-ops.
    interval_steps:
        Print a summary log line every *interval_steps* steps.
    output_file:
        Path to a JSONL file for full metric dumps.  *None* to disable.
    """

    def __init__(
        self,
        enabled: bool = True,
        interval_steps: int = 10,
        output_file: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._interval_steps = interval_steps
        self._history: deque[StepMetrics] = deque(maxlen=100)
        self._current: StepMetrics | None = None
        self._file_handle: IO[str] | None = None

        if output_file is not None and enabled:
            self._file_handle = open(output_file, "a")  # noqa: SIM115

    # ── step lifecycle ────────────────────────────────────────────────

    def begin_step(self, step: int) -> None:
        """Start collecting metrics for *step*."""
        if not self._enabled:
            return
        self._current = StepMetrics(step=step)

    def end_step(self) -> None:
        """Finalize the current step: archive, write JSONL, maybe log."""
        if not self._enabled or self._current is None:
            return

        metrics = self._current
        self._history.append(metrics)

        if self._file_handle is not None:
            self._write_jsonl(metrics)

        if self._interval_steps > 0 and metrics.step % self._interval_steps == 0:
            log.info(self._format_log_line(metrics))

        self._current = None

    # ── recording helpers ─────────────────────────────────────────────

    def record_h2d(self, size_bytes: int) -> None:
        """Record an H2D transfer of *size_bytes*."""
        if self._current is not None:
            self._current.h2d_bytes += size_bytes

    def record_d2h(self, size_bytes: int) -> None:
        """Record a D2H transfer of *size_bytes*."""
        if self._current is not None:
            self._current.d2h_bytes += size_bytes

    def record_stall(self, duration_ms: float) -> None:
        """Record a compute stall of *duration_ms*."""
        if self._current is not None:
            self._current.stall_time_ms += duration_ms
            self._current.stall_count += 1

    def record_eviction(self) -> None:
        """Record one eviction event."""
        if self._current is not None:
            self._current.evictions += 1

    def record_prefetch_hit(self) -> None:
        """Record a prefetch hit (block was GPU_READY when needed)."""
        if self._current is not None:
            self._current.prefetch_hits += 1

    def record_prefetch_miss(self) -> None:
        """Record a prefetch miss (block required sync wait)."""
        if self._current is not None:
            self._current.prefetch_misses += 1

    def record_nan_inf(self, nan_count: int, inf_count: int) -> None:
        """Record NaN/Inf counts detected by numeric guards."""
        if self._current is not None:
            self._current.nan_count += nan_count
            self._current.inf_count += inf_count

    def record_vram(self, used_mb: float, reserved_mb: float) -> None:
        """Record current VRAM utilization."""
        if self._current is not None:
            self._current.vram_used_mb = used_mb
            self._current.vram_reserved_mb = reserved_mb

    def record_pool_stats(self, free: int, in_use: int) -> None:
        """Record PinnedPool slab counts."""
        if self._current is not None:
            self._current.host_pool_free = free
            self._current.host_pool_in_use = in_use

    # ── rolling-window queries ────────────────────────────────────────

    def hit_rate(self) -> float:
        """Prefetch hit rate over the rolling window (0.0 .. 1.0)."""
        total_hits = sum(m.prefetch_hits for m in self._history)
        total_misses = sum(m.prefetch_misses for m in self._history)
        total = total_hits + total_misses
        if total == 0:
            return 0.0
        return total_hits / total

    def mean_stall_ms(self) -> float:
        """Mean stall_time_ms over the rolling window."""
        if not self._history:
            return 0.0
        return sum(m.stall_time_ms for m in self._history) / len(self._history)

    def max_stall_ms(self) -> float:
        """Max stall_time_ms in the rolling window."""
        if not self._history:
            return 0.0
        return max(m.stall_time_ms for m in self._history)

    def vram_trend(self) -> str:
        """Estimate VRAM trend from rolling window: growing, stable, or shrinking."""
        if len(self._history) < 3:
            return "stable"

        recent = list(self._history)
        n = len(recent)
        mid = n // 2

        first_half_avg = sum(m.vram_used_mb for m in recent[:mid]) / mid if mid > 0 else 0.0
        second_half_avg = (
            sum(m.vram_used_mb for m in recent[mid:]) / (n - mid) if (n - mid) > 0 else 0.0
        )

        delta = second_half_avg - first_half_avg
        # Threshold: 50 MB change = significant.
        if delta > 50.0:
            return "growing"
        elif delta < -50.0:
            return "shrinking"
        return "stable"

    # ── formatting ────────────────────────────────────────────────────

    def _format_log_line(self, metrics: StepMetrics) -> str:
        """Compact single-line summary matching spec format."""
        hr = self.hit_rate() * 100.0
        stall = metrics.stall_time_ms
        evict = metrics.evictions
        vram_gb = metrics.vram_used_mb / 1024.0
        pool_free = metrics.host_pool_free
        pool_total = metrics.host_pool_free + metrics.host_pool_in_use
        return (
            f"[STAGEHAND step={metrics.step}] "
            f"hit={hr:.1f}% "
            f"stall={stall:.1f}ms "
            f"evict={evict} "
            f"vram={vram_gb:.1f}G "
            f"pool={pool_free}/{pool_total}"
        )

    def _write_jsonl(self, metrics: StepMetrics) -> None:
        """Append one JSON line with all fields from *metrics*."""
        if self._file_handle is None:
            return
        self._file_handle.write(json.dumps(asdict(metrics)) + "\n")
        self._file_handle.flush()

    # ── cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the JSONL file handle."""
        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
