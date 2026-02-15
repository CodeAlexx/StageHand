"""Phase 2 acceptance tests — BlockRegistry, ResidencyMap, BudgetManager, NumericGuard.

All tests are runnable WITHOUT a GPU (CUDA calls are mocked).
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from stagehand.budget import BudgetManager
from stagehand.errors import (
    DtypeMismatchError,
    InvalidStateTransitionError,
    StagehandOOMError,
)
from stagehand.guards import NumericGuard
from stagehand.registry import BlockEntry, BlockRegistry
from stagehand.residency import (
    VALID_TRANSITIONS,
    BlockState,
    ResidencyEntry,
    ResidencyMap,
)
try:
    from serenity.training.adapters.lora import apply_lora
except ModuleNotFoundError:  # standalone package test environment
    apply_lora = None


# ── helpers ──────────────────────────────────────────────────────────────


def _make_mock_model(num_blocks: int = 4, in_features: int = 1024, out_features: int = 1024) -> nn.Module:
    """Build a simple model with named blocks that match a pattern.

    Structure::
        model.block.0 = Linear(in, out)
        model.block.1 = Linear(in, out)
        ...
    """
    blocks = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_blocks)])
    model = nn.Module()
    model.add_module("block", blocks)
    return model


def _build_registry(num_blocks: int = 4, dtype: torch.dtype = torch.bfloat16) -> BlockRegistry:
    """Helper: build + validate a registry from a mock model."""
    model = _make_mock_model(num_blocks)
    reg = BlockRegistry()
    reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=dtype)
    reg.validate(pool_capacity_bytes=1024 * 1024 * 1024)  # 1 GB
    return reg


# ── BlockRegistry tests ──────────────────────────────────────────────────


class TestBlockRegistry:
    """Tests for :class:`BlockRegistry`."""

    def test_build_block_count(self) -> None:
        """Registry contains exactly as many blocks as matched modules."""
        reg = _build_registry(num_blocks=6)
        assert len(reg) == 6

    def test_exec_order_sequential(self) -> None:
        """exec_order values are sequential starting from 0."""
        reg = _build_registry(num_blocks=5)
        orders = [e.exec_order for e in reg.blocks_in_order()]
        assert orders == list(range(5))

    def test_size_bytes_calculation(self) -> None:
        """size_bytes matches manual calculation: weight + bias in target dtype."""
        model = _make_mock_model(num_blocks=1, in_features=1024, out_features=1024)
        reg = BlockRegistry()
        reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=torch.bfloat16)

        entry = reg.blocks_in_order()[0]
        # Linear(1024, 1024): weight = 1024*1024 params, bias = 1024 params
        # bf16 = 2 bytes per element
        expected = (1024 * 1024 + 1024) * 2
        assert entry.size_bytes == expected

    def test_get_existing_block(self) -> None:
        """get() returns the correct entry by block_id."""
        reg = _build_registry(num_blocks=3)
        entry = reg.get("block.0")
        assert entry.block_id == "block.0"
        assert entry.group == "wan"

    def test_get_missing_block_raises(self) -> None:
        """get() raises KeyError for unknown block_id."""
        reg = _build_registry(num_blocks=2)
        with pytest.raises(KeyError):
            reg.get("nonexistent.block")

    def test_blocks_in_order_sorted(self) -> None:
        """blocks_in_order() is sorted by exec_order."""
        reg = _build_registry(num_blocks=4)
        entries = reg.blocks_in_order()
        for i in range(len(entries) - 1):
            assert entries[i].exec_order < entries[i + 1].exec_order

    def test_groups(self) -> None:
        """groups() returns correct grouping."""
        reg = _build_registry(num_blocks=3)
        grps = reg.groups()
        assert "wan" in grps
        assert len(grps["wan"]) == 3

    def test_validate_raises_for_oversized_block(self) -> None:
        """validate() raises StagehandOOMError when a block exceeds pool capacity."""
        model = _make_mock_model(num_blocks=1, in_features=1024, out_features=1024)
        reg = BlockRegistry()
        reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=torch.bfloat16)
        # Pool capacity of 1 byte — any block is too large
        with pytest.raises(StagehandOOMError, match="exceeds pool capacity"):
            reg.validate(pool_capacity_bytes=1)

    def test_frozen_after_validate(self) -> None:
        """Registry is frozen after validate(); further build_from_model raises."""
        reg = _build_registry(num_blocks=2)
        model = _make_mock_model(num_blocks=1)
        with pytest.raises(RuntimeError, match="frozen"):
            reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="dit", dtype=torch.bfloat16)

    def test_contains(self) -> None:
        """__contains__ works for block_ids."""
        reg = _build_registry(num_blocks=2)
        assert "block.0" in reg
        assert "block.1" in reg
        assert "block.99" not in reg

    def test_dtype_stored_correctly(self) -> None:
        """Entry dtype matches the dtype passed to build_from_model."""
        reg = _build_registry(num_blocks=1, dtype=torch.float16)
        entry = reg.blocks_in_order()[0]
        assert entry.dtype == torch.float16

    def test_module_ref_is_weakref(self) -> None:
        """module_ref is a weak reference that resolves to the original module."""
        model = _make_mock_model(num_blocks=1)
        reg = BlockRegistry()
        reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=torch.bfloat16)
        entry = reg.blocks_in_order()[0]
        # The weak reference should resolve while model is alive
        assert entry.module_ref() is not None

    def test_convert_to_file_backed_handles_lora_orig_param_names(self, tmp_path: Path) -> None:
        """LoRA-wrapped ``.orig`` params still map to base safetensors keys."""
        if apply_lora is None:
            pytest.skip("serenity lora adapter not available in standalone test env")

        class _LoRABlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = nn.Linear(8, 8, bias=False)
                self.to_k = nn.Linear(8, 8, bias=False)
                self.ff = nn.Linear(8, 8, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.to_q(x) + self.to_k(x))

        class _LoRAModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.ModuleList([_LoRABlock(), _LoRABlock()])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.block:
                    x = blk(x)
                return x

        model = _LoRAModel().cpu()
        ckpt_path = tmp_path / "base.safetensors"
        save_file(model.state_dict(), str(ckpt_path))

        # Wrapping linears introduces ".orig.weight" names in named_parameters().
        apply_lora(model, rank=2, alpha=2.0, target_modules=["to_q", "to_k", "ff"])

        reg = BlockRegistry()
        reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=torch.float32)
        reg.validate(pool_capacity_bytes=1024 * 1024 * 1024)

        converted = reg.convert_to_file_backed(str(ckpt_path))
        assert converted > 0

        entry = reg.get("block.0")
        assert entry.file_backed
        assert any(spec.param_name.endswith(".orig.weight") for spec in entry.file_param_specs)

        block0 = entry.module_ref()
        assert block0 is not None
        assert block0.to_q.orig.weight.numel() == 0

    def test_convert_to_file_backed_accepts_squareq_slab(self, tmp_path: Path) -> None:
        """SquareQ slab sources should convert frozen params to file-backed specs."""
        squareq_path = tmp_path / "tiny_squareq.fpk"
        payload = {
            "manifest": {
                "model_name": "tiny",
                "quant_version": "test",
                "layout": "rowwise_sym_int8",
                "pack_k": 1,
                "layers": [
                    {
                        "name": "block.0",
                        "out": 8,
                        "inp": 8,
                        "padded_in": 8,
                        "has_bias": True,
                    }
                ],
            },
            "layers": {
                "block.0": {
                    "qweight": torch.randint(-127, 128, (8, 8), dtype=torch.int8),
                    "scale": torch.ones(8, dtype=torch.float32),
                    "zero_point": torch.zeros(8, dtype=torch.float32),
                    "bias": torch.zeros(8, dtype=torch.float32),
                }
            },
        }
        torch.save(payload, squareq_path)

        model = _make_mock_model(num_blocks=1, in_features=8, out_features=8).cpu()
        for param in model.parameters():
            param.requires_grad_(False)
        reg = BlockRegistry()
        reg.build_from_model(model, block_pattern=r"^block\.\d+$", group="wan", dtype=torch.float32)
        reg.validate(pool_capacity_bytes=1024 * 1024 * 1024)

        converted = reg.convert_to_file_backed(str(squareq_path))
        assert converted == 2  # weight + bias

        entry = reg.get("block.0")
        assert entry.file_backed
        assert entry.squareq_backed
        assert entry.source_format == "squareq_bp8"
        assert len(entry.squareq_param_specs) == 2

        block = entry.module_ref()
        assert block is not None
        assert block.weight.numel() == 0
        assert block.bias is not None and block.bias.numel() == 0

    def test_candidate_tensor_keys_maps_wan_aliases(self) -> None:
        """WAN module names normalize to checkpoint key naming."""
        keys = BlockRegistry._candidate_tensor_keys("blocks.0", "attn1.to_q.orig.weight")
        assert "blocks.0.self_attn.q.weight" in keys

        keys = BlockRegistry._candidate_tensor_keys("blocks.0", "attn2.to_out.0.orig.bias")
        assert "blocks.0.cross_attn.o.bias" in keys

        keys = BlockRegistry._candidate_tensor_keys("blocks.0", "ffn.net.0.proj.orig.weight")
        assert "blocks.0.ffn.0.weight" in keys

        keys = BlockRegistry._candidate_tensor_keys("blocks.0", "scale_shift_table")
        assert "blocks.0.modulation" in keys


# ── ResidencyMap + BlockState tests ──────────────────────────────────────


class TestBlockState:
    """Tests for :class:`BlockState` and the state machine."""

    def test_all_valid_transitions_succeed(self) -> None:
        """Walk every legal transition path and verify no exception."""
        reg = _build_registry(num_blocks=1)
        bid = "block.0"

        # Path 1: UNLOADED -> HOST_STAGED -> PREFETCHING -> GPU_READY -> GPU_FREEING -> UNLOADED
        rmap = ResidencyMap(reg)
        rmap.transition(bid, BlockState.HOST_STAGED)
        assert rmap.get_state(bid) == BlockState.HOST_STAGED

        rmap.transition(bid, BlockState.PREFETCHING)
        assert rmap.get_state(bid) == BlockState.PREFETCHING

        rmap.transition(bid, BlockState.GPU_READY)
        assert rmap.get_state(bid) == BlockState.GPU_READY

        rmap.transition(bid, BlockState.GPU_FREEING)
        assert rmap.get_state(bid) == BlockState.GPU_FREEING

        rmap.transition(bid, BlockState.UNLOADED)
        assert rmap.get_state(bid) == BlockState.UNLOADED

    def test_eviction_path_to_host_staged(self) -> None:
        """GPU_READY -> EVICTING -> HOST_STAGED (D2H for blocks with gradients)."""
        reg = _build_registry(num_blocks=1)
        rmap = ResidencyMap(reg)
        bid = "block.0"

        rmap.transition(bid, BlockState.HOST_STAGED)
        rmap.transition(bid, BlockState.PREFETCHING)
        rmap.transition(bid, BlockState.GPU_READY)
        rmap.transition(bid, BlockState.EVICTING)
        assert rmap.get_state(bid) == BlockState.EVICTING

        rmap.transition(bid, BlockState.HOST_STAGED)
        assert rmap.get_state(bid) == BlockState.HOST_STAGED

    def test_eviction_path_to_unloaded(self) -> None:
        """GPU_READY -> EVICTING -> UNLOADED (D2H completed, slab released)."""
        reg = _build_registry(num_blocks=1)
        rmap = ResidencyMap(reg)
        bid = "block.0"

        rmap.transition(bid, BlockState.HOST_STAGED)
        rmap.transition(bid, BlockState.PREFETCHING)
        rmap.transition(bid, BlockState.GPU_READY)
        rmap.transition(bid, BlockState.EVICTING)
        rmap.transition(bid, BlockState.UNLOADED)
        assert rmap.get_state(bid) == BlockState.UNLOADED

    def test_all_invalid_transitions_raise(self) -> None:
        """Every transition NOT in VALID_TRANSITIONS raises InvalidStateTransitionError."""
        all_states = list(BlockState)
        reg = _build_registry(num_blocks=1)
        bid = "block.0"

        for src in all_states:
            allowed = VALID_TRANSITIONS.get(src, frozenset())
            for dst in all_states:
                if dst in allowed:
                    continue
                # Create a fresh map and force the source state
                rmap = ResidencyMap(reg)
                rmap.get_entry(bid).state = src
                with pytest.raises(InvalidStateTransitionError):
                    rmap.transition(bid, dst)


class TestResidencyMap:
    """Tests for :class:`ResidencyMap`."""

    def test_initial_state_is_unloaded(self) -> None:
        """All blocks start in UNLOADED state."""
        reg = _build_registry(num_blocks=3)
        rmap = ResidencyMap(reg)
        for entry in [rmap.get_entry("block.0"), rmap.get_entry("block.1"), rmap.get_entry("block.2")]:
            assert entry.state == BlockState.UNLOADED

    def test_refcount_prevents_eviction(self) -> None:
        """can_evict returns False when refcount > 0."""
        reg = _build_registry(num_blocks=1)
        rmap = ResidencyMap(reg)
        bid = "block.0"

        # Move to GPU_READY
        rmap.transition(bid, BlockState.HOST_STAGED)
        rmap.transition(bid, BlockState.PREFETCHING)
        rmap.transition(bid, BlockState.GPU_READY)

        # Without refcount, can evict
        assert rmap.can_evict(bid) is True

        # With refcount, cannot evict
        rmap.increment_ref(bid)
        assert rmap.can_evict(bid) is False

        # Release ref, can evict again
        rmap.decrement_ref(bid)
        assert rmap.can_evict(bid) is True

    def test_decrement_ref_below_zero_raises(self) -> None:
        """Decrementing refcount below zero raises ValueError."""
        reg = _build_registry(num_blocks=1)
        rmap = ResidencyMap(reg)
        with pytest.raises(ValueError):
            rmap.decrement_ref("block.0")

    def test_gpu_resident_blocks(self) -> None:
        """gpu_resident_blocks returns only blocks in GPU_READY state."""
        reg = _build_registry(num_blocks=3)
        rmap = ResidencyMap(reg)

        # Move block.0 and block.1 to GPU_READY
        for bid in ["block.0", "block.1"]:
            rmap.transition(bid, BlockState.HOST_STAGED)
            rmap.transition(bid, BlockState.PREFETCHING)
            rmap.transition(bid, BlockState.GPU_READY)

        resident = rmap.gpu_resident_blocks()
        assert sorted(resident) == ["block.0", "block.1"]

    def test_eviction_candidates_respects_cooldown(self) -> None:
        """eviction_candidates only returns blocks outside the cooldown window."""
        reg = _build_registry(num_blocks=3)
        rmap = ResidencyMap(reg)

        # Move all blocks to GPU_READY
        for bid in ["block.0", "block.1", "block.2"]:
            rmap.transition(bid, BlockState.HOST_STAGED)
            rmap.transition(bid, BlockState.PREFETCHING)
            rmap.transition(bid, BlockState.GPU_READY)

        # Set last_used_step: block.0=step 5, block.1=step 8, block.2=step 10
        rmap.get_entry("block.0").last_used_step = 5
        rmap.get_entry("block.1").last_used_step = 8
        rmap.get_entry("block.2").last_used_step = 10

        # At step 10, cooldown=2: only blocks with last_used <= 8 qualify
        candidates = rmap.eviction_candidates(current_step=10, cooldown_steps=2)
        candidate_ids = sorted(bid for bid, _ in candidates)
        assert candidate_ids == ["block.0", "block.1"]

    def test_eviction_candidates_excludes_refcounted(self) -> None:
        """Blocks with refcount > 0 are never eviction candidates."""
        reg = _build_registry(num_blocks=2)
        rmap = ResidencyMap(reg)

        for bid in ["block.0", "block.1"]:
            rmap.transition(bid, BlockState.HOST_STAGED)
            rmap.transition(bid, BlockState.PREFETCHING)
            rmap.transition(bid, BlockState.GPU_READY)
            rmap.get_entry(bid).last_used_step = 0

        rmap.increment_ref("block.0")
        candidates = rmap.eviction_candidates(current_step=100, cooldown_steps=0)
        candidate_ids = [bid for bid, _ in candidates]
        assert "block.0" not in candidate_ids
        assert "block.1" in candidate_ids

    def test_len_matches_registry(self) -> None:
        """ResidencyMap length matches the registry block count."""
        reg = _build_registry(num_blocks=5)
        rmap = ResidencyMap(reg)
        assert len(rmap) == 5

    def test_contains(self) -> None:
        """__contains__ works for known and unknown block_ids."""
        reg = _build_registry(num_blocks=2)
        rmap = ResidencyMap(reg)
        assert "block.0" in rmap
        assert "nonexistent" not in rmap


# ── BudgetManager tests ──────────────────────────────────────────────────


class TestBudgetManager:
    """Tests for :class:`BudgetManager`."""

    def test_invalid_watermarks_raise(self) -> None:
        """high_watermark_mb must be greater than low_watermark_mb."""
        with pytest.raises(ValueError, match="must be greater"):
            BudgetManager(high_watermark_mb=1000, low_watermark_mb=2000)
        with pytest.raises(ValueError, match="must be greater"):
            BudgetManager(high_watermark_mb=1000, low_watermark_mb=1000)

    def test_above_high_watermark(self) -> None:
        """above_high_watermark returns True when usage exceeds threshold."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        # Mock memory_allocated to return 120 MB worth of bytes
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_allocated", return_value=120 * 1024 * 1024):
            assert bm.above_high_watermark() is True
            assert bm.should_evict() is True
            assert bm.can_prefetch() is False

    def test_below_low_watermark(self) -> None:
        """below_low_watermark returns True when usage is under threshold."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_allocated", return_value=30 * 1024 * 1024):
            assert bm.below_low_watermark() is True
            assert bm.above_high_watermark() is False
            assert bm.can_prefetch() is True

    def test_between_watermarks(self) -> None:
        """Between low and high watermarks: normal ops, can prefetch, no eviction."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_allocated", return_value=75 * 1024 * 1024):
            assert bm.below_low_watermark() is False
            assert bm.above_high_watermark() is False
            assert bm.can_prefetch() is True
            assert bm.should_evict() is False

    def test_headroom_mb(self) -> None:
        """headroom_mb returns correct remaining space."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_allocated", return_value=60 * 1024 * 1024):
            assert bm.headroom_mb() == pytest.approx(40.0)

    def test_no_cuda_returns_zero(self) -> None:
        """Without CUDA, vram_used_mb and vram_reserved_mb return 0."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert bm.vram_used_mb() == 0.0
            assert bm.vram_reserved_mb() == 0.0
            # 0 < 100 so we are below high watermark
            assert bm.can_prefetch() is True
            assert bm.should_evict() is False

    def test_vram_reserved_mb(self) -> None:
        """vram_reserved_mb wraps torch.cuda.memory_reserved."""
        bm = BudgetManager(high_watermark_mb=100, low_watermark_mb=50)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_reserved", return_value=200 * 1024 * 1024):
            assert bm.vram_reserved_mb() == pytest.approx(200.0)


# ── NumericGuard tests ───────────────────────────────────────────────────


class TestNumericGuard:
    """Tests for :class:`NumericGuard`."""

    def test_strict_dtype_check_passes(self) -> None:
        """No error when dtype matches expected."""
        guard = NumericGuard(strict_bf16=True)
        t = torch.zeros(4, dtype=torch.bfloat16)
        guard.check_dtype(t, torch.bfloat16, context="test_block")

    def test_strict_dtype_check_fails_on_f32(self) -> None:
        """Strict mode catches fp32 tensor when bf16 is expected."""
        guard = NumericGuard(strict_bf16=True)
        t = torch.zeros(4, dtype=torch.float32)
        with pytest.raises(DtypeMismatchError, match="expected.*bfloat16.*got.*float32"):
            guard.check_dtype(t, torch.bfloat16, context="test_block")

    def test_permissive_dtype_check_allows_mismatch(self) -> None:
        """Permissive mode (strict_bf16=False) does not raise on mismatch."""
        guard = NumericGuard(strict_bf16=False)
        t = torch.zeros(4, dtype=torch.float32)
        guard.check_dtype(t, torch.bfloat16, context="test_block")  # no error

    def test_nan_detection(self) -> None:
        """check_output detects NaN values."""
        guard = NumericGuard(nan_inf_check=True)
        t = torch.tensor([1.0, float("nan"), 3.0])
        nan_count, inf_count = guard.check_output(t, block_id="wan.spatial.0", step=42)
        assert nan_count == 1
        assert inf_count == 0

    def test_inf_detection(self) -> None:
        """check_output detects Inf values."""
        guard = NumericGuard(nan_inf_check=True)
        t = torch.tensor([1.0, float("inf"), float("-inf"), 4.0])
        nan_count, inf_count = guard.check_output(t, block_id="wan.temporal.3", step=7)
        assert nan_count == 0
        assert inf_count == 2

    def test_mixed_nan_inf_detection(self) -> None:
        """check_output counts NaN and Inf separately."""
        guard = NumericGuard(nan_inf_check=True)
        t = torch.tensor([float("nan"), float("inf"), float("nan"), 1.0])
        nan_count, inf_count = guard.check_output(t, block_id="block.0", step=0)
        assert nan_count == 2
        assert inf_count == 1

    def test_clean_tensor_returns_zero(self) -> None:
        """check_output returns (0, 0) for a clean tensor."""
        guard = NumericGuard(nan_inf_check=True)
        t = torch.randn(100)
        nan_count, inf_count = guard.check_output(t, block_id="block.0", step=0)
        assert nan_count == 0
        assert inf_count == 0

    def test_nan_inf_check_disabled(self) -> None:
        """With nan_inf_check=False, always returns (0, 0)."""
        guard = NumericGuard(nan_inf_check=False)
        t = torch.tensor([float("nan"), float("inf")])
        nan_count, inf_count = guard.check_output(t, block_id="block.0", step=0)
        assert nan_count == 0
        assert inf_count == 0

    def test_promotion_detection_bf16_to_f32(self) -> None:
        """check_promotion detects bf16 -> f32 promotion."""
        guard = NumericGuard(fail_on_dtype_promotion=True)
        with pytest.raises(DtypeMismatchError, match="promotion"):
            guard.check_promotion(torch.bfloat16, torch.float32, context="test")

    def test_promotion_detection_f16_to_f32(self) -> None:
        """check_promotion detects f16 -> f32 promotion."""
        guard = NumericGuard(fail_on_dtype_promotion=True)
        with pytest.raises(DtypeMismatchError, match="promotion"):
            guard.check_promotion(torch.float16, torch.float32, context="test")

    def test_no_promotion_same_dtype(self) -> None:
        """Same input/output dtype is not a promotion."""
        guard = NumericGuard(fail_on_dtype_promotion=True)
        guard.check_promotion(torch.bfloat16, torch.bfloat16, context="test")

    def test_no_promotion_downcast(self) -> None:
        """f32 -> bf16 is not a promotion (downcast)."""
        guard = NumericGuard(fail_on_dtype_promotion=True)
        guard.check_promotion(torch.float32, torch.bfloat16, context="test")

    def test_permissive_promotion_allows_all(self) -> None:
        """With fail_on_dtype_promotion=False, no error on promotion."""
        guard = NumericGuard(fail_on_dtype_promotion=False)
        guard.check_promotion(torch.bfloat16, torch.float32, context="test")

    def test_same_precision_level_no_error(self) -> None:
        """bf16 -> f16 (same precision level) is not flagged as promotion."""
        guard = NumericGuard(fail_on_dtype_promotion=True)
        guard.check_promotion(torch.bfloat16, torch.float16, context="test")
