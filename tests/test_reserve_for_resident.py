"""Tests for StagehandRuntime.reserve_for_resident and as_guest."""
from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from stagehand import (
    StagehandConfig,
    StagehandRuntime,
    ResidentPriority,
)
from stagehand.budget import BudgetManager
from stagehand.residency import BlockState


# ── helpers ──────────────────────────────────────────────────────────────


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoBlockModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.block = nn.ModuleList([TinyBlock(dim), TinyBlock(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block:
            x = b(x)
        return x


def _make_runtime(model: nn.Module | None = None, **kwargs) -> StagehandRuntime:
    if model is None:
        model = TwoBlockModel()
    cfg = StagehandConfig(
        pinned_pool_mb=8,
        pinned_slab_mb=1,
        vram_high_watermark_mb=22000,
        vram_low_watermark_mb=18000,
        telemetry_enabled=False,
        **kwargs,
    )
    return StagehandRuntime(
        model=model,
        config=cfg,
        block_pattern=r"block\.\d+$",
        group="test",
        dtype=torch.float32,
        inference_mode=True,
    )


# ── BudgetManager reservation tests ─────────────────────────────────────


class TestBudgetManagerReservation:
    """Unit tests for BudgetManager reserve_bytes / release_reserved_bytes."""

    def test_reserve_increases_reserved(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        assert bm.reserved_bytes == 0
        bm.reserve_bytes(1_000_000, label="test")
        assert bm.reserved_bytes == 1_000_000

    def test_release_decreases_reserved(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        bm.reserve_bytes(5_000_000)
        bm.release_reserved_bytes(3_000_000)
        assert bm.reserved_bytes == 2_000_000

    def test_release_clamps_to_zero(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        bm.reserve_bytes(100)
        bm.release_reserved_bytes(999)
        assert bm.reserved_bytes == 0

    def test_can_guest_allocate_with_headroom(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        # Without reservation, small allocations should fit
        assert bm.can_guest_allocate(100)

    def test_can_guest_allocate_denied_after_reservation(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        # Reserve nearly the entire budget
        budget_bytes = 22000 * 1024 * 1024
        bm.reserve_bytes(budget_bytes)
        # Now even a small allocation should be denied
        assert not bm.can_guest_allocate(1024 * 1024)

    def test_repr_includes_reserved(self) -> None:
        bm = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        bm.reserve_bytes(1024 * 1024 * 1024)  # 1GB
        r = repr(bm)
        assert "reserved=" in r


# ── reserve_for_resident tests ───────────────────────────────────────────


class TestReserveForResident:
    """Tests for StagehandRuntime.reserve_for_resident."""

    def test_basic_reservation(self) -> None:
        rt = _make_runtime()
        total_size = sum(e.size_bytes for e in rt._registry.blocks_in_order())

        rt.reserve_for_resident(priority=ResidentPriority.PRIMARY)

        assert rt._budget.reserved_bytes == total_size
        assert hasattr(rt, "_reservations")
        assert id(rt) in rt._reservations

    def test_primary_protects_blocks(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        rt.reserve_for_resident(priority=ResidentPriority.PRIMARY)

        # Blocks should be protected
        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 0

    def test_guest_does_not_protect_blocks(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        rt.reserve_for_resident(priority=ResidentPriority.GUEST)

        # GUEST blocks should still be eviction candidates
        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 2


class TestReleaseReservation:
    """Tests for StagehandRuntime.release_reservation."""

    def test_release_restores_budget(self) -> None:
        rt = _make_runtime()
        total_size = sum(e.size_bytes for e in rt._registry.blocks_in_order())

        rt.reserve_for_resident(priority=ResidentPriority.PRIMARY)
        assert rt._budget.reserved_bytes == total_size

        rt.release_reservation()
        assert rt._budget.reserved_bytes == 0

    def test_release_unprotects_blocks(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        rt.reserve_for_resident(priority=ResidentPriority.PRIMARY)
        assert len(residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 0

        rt.release_reservation()
        assert len(residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 2

    def test_release_nonexistent_is_noop(self) -> None:
        rt = _make_runtime()
        rt.release_reservation()  # Should not raise


# ── as_guest tests ───────────────────────────────────────────────────────


class TestAsGuest:
    """Tests for StagehandRuntime.as_guest context manager."""

    def test_guest_reservation_released_on_exit(self) -> None:
        host_rt = _make_runtime()
        guest_model = TwoBlockModel(dim=8)
        guest_rt = _make_runtime(model=guest_model)

        guest_size = sum(e.size_bytes for e in guest_rt._registry.blocks_in_order())

        with host_rt.as_guest(guest_rt):
            assert host_rt._budget.reserved_bytes == guest_size

        # Reservation released on context exit
        assert host_rt._budget.reserved_bytes == 0

    def test_guest_exception_still_cleans_up(self) -> None:
        host_rt = _make_runtime()
        guest_model = TwoBlockModel(dim=8)
        guest_rt = _make_runtime(model=guest_model)

        with pytest.raises(RuntimeError, match="guest error"):
            with host_rt.as_guest(guest_rt):
                raise RuntimeError("guest error")

        # Must be cleaned up despite exception
        assert host_rt._budget.reserved_bytes == 0

    def test_prefetch_window_restored_on_exit(self) -> None:
        host_rt = _make_runtime()
        original_window = host_rt._scheduler._policy.prefetch_window

        guest_model = TwoBlockModel(dim=8)
        guest_rt = _make_runtime(model=guest_model)

        # Force tight headroom by reserving nearly everything
        budget_bytes = 22000 * 1024 * 1024
        host_rt._budget.reserve_bytes(budget_bytes - 100)

        with host_rt.as_guest(guest_rt):
            # Prefetch window should be reduced
            assert host_rt._scheduler._policy.prefetch_window == 1

        # Must be restored
        host_rt._budget.release_reserved_bytes(budget_bytes - 100)
        assert host_rt._scheduler._policy.prefetch_window == original_window


# ── combined scenario test ───────────────────────────────────────────────


class TestCombinedScenario:
    """Integration test: PRIMARY reservation + guest loads."""

    def test_primary_survives_guest_lifecycle(self) -> None:
        # Create "DiT" runtime (primary)
        dit_model = TwoBlockModel(dim=32)
        dit_rt = _make_runtime(model=dit_model)
        dit_residency = dit_rt._residency

        # Set DiT blocks to GPU_READY
        for entry in dit_rt._registry.blocks_in_order():
            bid = entry.block_id
            dit_residency.transition(bid, BlockState.HOST_STAGED)
            dit_residency.transition(bid, BlockState.PREFETCHING)
            dit_residency.transition(bid, BlockState.GPU_READY)
            dit_residency.get_entry(bid).last_used_step = 0

        dit_size = sum(e.size_bytes for e in dit_rt._registry.blocks_in_order())

        # Reserve DiT as PRIMARY
        dit_rt.reserve_for_resident(priority=ResidentPriority.PRIMARY)
        assert dit_rt._budget.reserved_bytes == dit_size

        # DiT blocks should be protected
        assert len(dit_residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 0

        # Load guest 1 (spatial upscaler)
        guest1_model = TwoBlockModel(dim=8)
        guest1_rt = _make_runtime(model=guest1_model)
        guest1_size = sum(e.size_bytes for e in guest1_rt._registry.blocks_in_order())

        with dit_rt.as_guest(guest1_rt):
            assert dit_rt._budget.reserved_bytes == dit_size + guest1_size
            # DiT blocks still protected
            assert len(dit_residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 0

        # After guest1 exits, only DiT reservation remains
        assert dit_rt._budget.reserved_bytes == dit_size

        # Load guest 2 (temporal upscaler)
        guest2_model = TwoBlockModel(dim=8)
        guest2_rt = _make_runtime(model=guest2_model)

        with dit_rt.as_guest(guest2_rt):
            # DiT blocks still protected
            assert len(dit_residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 0

        # Release DiT reservation
        dit_rt.release_reservation()
        assert dit_rt._budget.reserved_bytes == 0

        # DiT blocks should now be evictable
        assert len(dit_residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 2
