"""Stagehand runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["StagehandConfig"]


@dataclass
class StagehandConfig:
    """All configuration knobs for the stagehand block-swapping runtime.

    Defaults match Section 3 of the spec (3090 24 GB / 64 GB RAM target).
    """

    # ── master switch ────────────────────────────────────────────────
    stagehand_enabled: bool = True

    # ── memory ───────────────────────────────────────────────────────
    pinned_pool_mb: int = 8192
    pinned_slab_mb: int = 256
    pinned_slab_alignment: int = 512  # byte alignment for DMA
    vram_high_watermark_mb: int = 22000
    vram_low_watermark_mb: int = 18000

    # ── transfer ─────────────────────────────────────────────────────
    max_inflight_transfers: int = 2
    copy_stream_priority: int = -1

    # ── scheduler / policy ───────────────────────────────────────────
    policy: str = "static"
    prefetch_window_blocks: int = 3
    eviction_cooldown_steps: int = 2

    # ── numeric guards ───────────────────────────────────────────────
    strict_bf16: bool = True
    fail_on_dtype_promotion: bool = True
    nan_inf_check: bool = True

    # ── telemetry ────────────────────────────────────────────────────
    telemetry_enabled: bool = True
    telemetry_interval_steps: int = 10
    telemetry_file: str = "stagehand_telemetry.jsonl"

    # ── debug ────────────────────────────────────────────────────────
    debug_trace: bool = False
