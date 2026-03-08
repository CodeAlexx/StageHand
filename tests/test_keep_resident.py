"""Tests for StagehandRuntime.keep_resident context manager."""
from __future__ import annotations

import logging
import re

import pytest
import torch
from torch import nn

from stagehand import (
    BlockRegistry,
    ResidencyEntry,
    ResidencyMap,
    StagehandConfig,
    StagehandRuntime,
)
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


# ── tests ────────────────────────────────────────────────────────────────


class TestKeepResidentBasic:
    """Basic protection: blocks are not eviction candidates inside the context."""

    def test_protection_active_inside_context(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        # Manually set blocks to GPU_READY with refcount 0 so they'd normally
        # be eviction candidates.
        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            re_entry = residency.get_entry(bid)
            re_entry.last_used_step = 0

        # Without protection: should have 2 candidates
        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 2

        # With protection: should have 0 candidates
        with rt.keep_resident():
            candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
            assert len(candidates) == 0

        # After context: back to 2 candidates
        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 2


class TestKeepResidentContextExit:
    """Blocks become normal eviction candidates after context exits."""

    def test_protection_released_on_exit(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        with rt.keep_resident():
            pass

        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 2

    def test_can_evict_after_context(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        with rt.keep_resident():
            for entry in rt._registry.blocks_in_order():
                assert not residency.can_evict(entry.block_id)

        for entry in rt._registry.blocks_in_order():
            assert residency.can_evict(entry.block_id)


class TestKeepResidentExceptionSafety:
    """Protection is released even if an exception occurs inside the context."""

    def test_exception_releases_protection(self) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        with pytest.raises(RuntimeError, match="test exception"):
            with rt.keep_resident():
                assert len(residency.eviction_candidates(current_step=10, cooldown_steps=0)) == 0
                raise RuntimeError("test exception")

        # Protection must be released despite the exception
        candidates = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(candidates) == 2
        assert len(residency._protected_blocks) == 0


class TestKeepResidentLastResort:
    """When ONLY protected blocks remain and VRAM is critical, evict with warning."""

    def test_last_resort_eviction_logs_warning(self, caplog) -> None:
        rt = _make_runtime()
        residency = rt._residency

        for entry in rt._registry.blocks_in_order():
            bid = entry.block_id
            residency.transition(bid, BlockState.HOST_STAGED)
            residency.transition(bid, BlockState.PREFETCHING)
            residency.transition(bid, BlockState.GPU_READY)
            residency.get_entry(bid).last_used_step = 0

        # protected_eviction_candidates should return them sorted by LRU
        residency.protect_blocks({entry.block_id for entry in rt._registry.blocks_in_order()})
        protected = residency.protected_eviction_candidates()
        assert len(protected) == 2

        # Normal candidates should be empty
        normal = residency.eviction_candidates(current_step=10, cooldown_steps=0)
        assert len(normal) == 0


class TestKeepResidentNoBlocks:
    """Calling keep_resident with no blocks logs a warning and doesn't crash."""

    def test_no_blocks_warning(self, caplog) -> None:
        # Create a model with no matching blocks
        model = nn.Linear(4, 4)
        cfg = StagehandConfig(
            pinned_pool_mb=8,
            pinned_slab_mb=1,
            vram_high_watermark_mb=22000,
            vram_low_watermark_mb=18000,
            telemetry_enabled=False,
        )
        rt = StagehandRuntime(
            model=model,
            config=cfg,
            block_pattern=r"nonexistent_pattern_xyz",
            group="test",
            dtype=torch.float32,
            inference_mode=True,
        )

        with caplog.at_level(logging.WARNING):
            with rt.keep_resident():
                pass  # Should not raise

        assert any("no blocks found" in record.message for record in caplog.records)


class TestKeepResidentOtherModelsUnaffected:
    """Prefetching for other models still works during keep_resident."""

    def test_only_target_blocks_protected(self) -> None:
        # Create two separate runtimes
        model1 = TwoBlockModel(dim=16)
        model2 = TwoBlockModel(dim=16)
        rt1 = _make_runtime(model=model1)
        rt2 = _make_runtime(model=model2)

        residency1 = rt1._residency
        residency2 = rt2._residency

        # Set all blocks in both runtimes to GPU_READY
        for rt, residency in [(rt1, residency1), (rt2, residency2)]:
            for entry in rt._registry.blocks_in_order():
                bid = entry.block_id
                residency.transition(bid, BlockState.HOST_STAGED)
                residency.transition(bid, BlockState.PREFETCHING)
                residency.transition(bid, BlockState.GPU_READY)
                residency.get_entry(bid).last_used_step = 0

        # Protect rt1's blocks only
        with rt1.keep_resident():
            # rt1's blocks should be protected
            c1 = residency1.eviction_candidates(current_step=10, cooldown_steps=0)
            assert len(c1) == 0

            # rt2's blocks should NOT be affected (separate residency map)
            c2 = residency2.eviction_candidates(current_step=10, cooldown_steps=0)
            assert len(c2) == 2
