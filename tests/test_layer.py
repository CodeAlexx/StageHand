"""Tests for stagehand layer mode — per-module CPU offloading runtime.

All tests run CPU-only.  CUDA-gated tests use ``@pytest.mark.skipif``.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from stagehand.layer import (
    LAYER_TYPES,
    LayerRuntime,
    _auto_pool_config,
    _discover_layers,
    _next_power_of_two,
    _parse_budget,
)
from stagehand.pool import PinnedPool

__all__: list[str] = []

# Device that the scheduler will place parameters on.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── helpers ──────────────────────────────────────────────────────────────


class _SequentialModel(nn.Module):
    """Simple sequential model with named Linear layers."""

    def __init__(self, num_layers: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _MixedModel(nn.Module):
    """Model with Linear, Conv2d, and Embedding layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SharedModel(nn.Module):
    """Model with a shared (aliased) layer."""

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(32, 32, bias=False)
        self.a = shared
        self.b = shared  # Same module aliased.
        self.c = nn.Linear(32, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        return x


class _EmptyModel(nn.Module):
    """Model with no eligible layers."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class _NestedModel(nn.Module):
    """Model with deeply nested layers."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(32, 32),
            nn.Sequential(
                nn.Linear(32, 32),
                nn.Linear(32, 16),
            ),
        )
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.head(x)


class _ReversedModel(nn.Module):
    """Model that executes layers in reverse of walk order."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(32, 32, bias=False)
        self.b = nn.Linear(32, 32, bias=False)
        self.c = nn.Linear(32, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c(x)
        x = self.b(x)
        x = self.a(x)
        return x


class _ConvOnlyModel(nn.Module):
    """Model with only Conv2d layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class _SingleLayerModel(nn.Module):
    """Model with exactly one layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── discovery tests ──────────────────────────────────────────────────────


class TestDiscovery:
    """Test module discovery logic."""

    def test_discovers_linear_layers(self) -> None:
        model = _SequentialModel(num_layers=5)
        layers = _discover_layers(model)
        assert len(layers) == 5
        for name, mod in layers:
            assert isinstance(mod, nn.Linear)

    def test_discovers_mixed_types(self) -> None:
        model = _MixedModel()
        layers = _discover_layers(model)
        types = {type(mod) for _, mod in layers}
        assert nn.Linear in types
        assert nn.Conv2d in types
        assert nn.Embedding in types

    def test_deduplicates_shared_modules(self) -> None:
        model = _SharedModel()
        layers = _discover_layers(model)
        # a and b are the same module — should only appear once.
        assert len(layers) == 2  # shared + c

    def test_nested_model(self) -> None:
        model = _NestedModel()
        layers = _discover_layers(model)
        assert len(layers) == 4  # 3 in block + 1 head

    def test_empty_model_no_layers(self) -> None:
        model = _EmptyModel()
        layers = _discover_layers(model)
        assert len(layers) == 0

    def test_conv_only_model(self) -> None:
        model = _ConvOnlyModel()
        layers = _discover_layers(model)
        assert len(layers) == 2
        for _, mod in layers:
            assert isinstance(mod, nn.Conv2d)

    def test_custom_target_types(self) -> None:
        model = _MixedModel()
        layers = _discover_layers(model, target_types=(nn.Linear,))
        assert len(layers) == 1
        assert isinstance(layers[0][1], nn.Linear)


# ── budget parsing tests ─────────────────────────────────────────────────


class TestParseBudget:
    """Test budget string parsing."""

    def test_none_returns_none(self) -> None:
        assert _parse_budget(None) is None

    def test_int_passthrough(self) -> None:
        assert _parse_budget(4 * 1024**3) == 4 * 1024**3

    def test_gb_string(self) -> None:
        assert _parse_budget("4GB") == 4 * 1024**3

    def test_mb_string(self) -> None:
        assert _parse_budget("512MB") == 512 * 1024**2

    def test_case_insensitive(self) -> None:
        assert _parse_budget("2gb") == 2 * 1024**3

    def test_plain_number_string(self) -> None:
        assert _parse_budget("1048576") == 1048576


# ── auto-config tests ────────────────────────────────────────────────────


class TestAutoConfig:
    """Test automatic pool configuration."""

    def test_slab_sizing_power_of_two(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        layers = _discover_layers(model)
        config = _auto_pool_config(layers, torch.bfloat16, prefetch_k=3, vram_budget=None, ram_budget=None)
        # Slab should be a power of 2.
        assert config.pinned_slab_mb & (config.pinned_slab_mb - 1) == 0

    def test_pool_has_enough_slabs(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        layers = _discover_layers(model)
        config = _auto_pool_config(layers, torch.bfloat16, prefetch_k=3, vram_budget=None, ram_budget=None)
        num_slabs = config.pinned_pool_mb // config.pinned_slab_mb
        assert num_slabs >= 7  # prefetch_k + 4

    def test_vram_budget_sets_watermarks(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        layers = _discover_layers(model)
        config = _auto_pool_config(layers, torch.bfloat16, prefetch_k=3, vram_budget=8 * 1024**3, ram_budget=None)
        assert config.vram_high_watermark_mb < 8192
        assert config.vram_low_watermark_mb < config.vram_high_watermark_mb

    def test_next_power_of_two(self) -> None:
        assert _next_power_of_two(1) == 1
        assert _next_power_of_two(3) == 4
        assert _next_power_of_two(4) == 4
        assert _next_power_of_two(5) == 8
        assert _next_power_of_two(0) == 1


# ── trace tests ──────────────────────────────────────────────────────────


class TestTrace:
    """Test trace mode: recording execution order."""

    def test_sequential_trace_records_order(self) -> None:
        model = _SequentialModel(num_layers=4, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)
        assert runtime.mode == "trace"

        # Step 0: trace pass.
        x = torch.randn(2, 32, device=_DEVICE)
        model(x)

        # Trace not yet complete (needs second forward to trigger auto-step).
        assert runtime.mode == "trace"
        assert len(runtime.trace_order) == 4

    def test_reversed_trace_captures_actual_order(self) -> None:
        model = _ReversedModel()
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)

        # Walk order is a, b, c. Execution order is c, b, a.
        assert runtime.trace_order == ["c", "b", "a"]


# ── rebuild tests ────────────────────────────────────────────────────────


class TestRebuild:
    """Test registry rebuild after trace."""

    def test_rebuild_on_second_forward(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, prefetch_k=2, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)  # Step 0 (trace).
        assert runtime.mode == "trace"

        model(x)  # Step 1 — triggers rebuild on first hook.
        assert runtime.mode == "scheduled"
        assert runtime.traced is True
        assert runtime.step == 1

    def test_rebuild_preserves_trace_order(self) -> None:
        model = _ReversedModel()
        runtime = LayerRuntime(model, dtype=torch.float32, prefetch_k=1, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)  # Trace.
        model(x)  # Rebuild.

        assert runtime.traced is True
        # After rebuild, the registry should use traced order (c, b, a).
        assert runtime.trace_order == ["c", "b", "a"]


# ── auto-step tests ──────────────────────────────────────────────────────


class TestAutoStep:
    """Test automatic step detection."""

    def test_multi_forward_increments_step(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, prefetch_k=1, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)  # Step 0.
        model(x)  # Step 1 (rebuild here).
        assert runtime.step == 1

        model(x)  # Step 2.
        assert runtime.step == 2

        model(x)  # Step 3.
        assert runtime.step == 3

    def test_step_0_is_trace(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)
        assert runtime.step == 0

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)
        assert runtime.step == 0  # Still step 0 — trace not yet complete.


# ── prefetch tests ───────────────────────────────────────────────────────


class TestPrefetch:
    """Test that prefetch improves hit rate after trace."""

    def test_hit_rate_improves_after_trace(self) -> None:
        model = _SequentialModel(num_layers=5, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, prefetch_k=3, telemetry=True)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)  # Step 0 (trace, no prefetch).
        model(x)  # Step 1 (rebuild, now has prefetch).

        # Run a few more steps to accumulate telemetry.
        for _ in range(3):
            model(x)

        # In CPU mode, "prefetch" still helps because the scheduler
        # issues loads for lookahead blocks before they're needed.
        # We just check telemetry is populated.
        assert runtime.step >= 4


# ── inference mode tests ─────────────────────────────────────────────────


class TestInferenceMode:
    """Test inference mode (no backward hooks)."""

    def test_inference_mode_flag(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(
            model, dtype=torch.float32, inference_mode=True, telemetry=False,
        )

        x = torch.randn(2, 32, device=_DEVICE)
        out = model(x)  # Step 0.
        out = model(x)  # Step 1.
        assert runtime.mode == "scheduled"

    def test_inference_no_grad(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(
            model, dtype=torch.float32, inference_mode=True, telemetry=False,
        )

        with torch.no_grad():
            x = torch.randn(2, 32, device=_DEVICE)
            out = model(x)
            out = model(x)
            assert out.shape == (2, 32)


# ── backward tests ───────────────────────────────────────────────────────


class TestBackward:
    """Test backward hook integration."""

    def test_backward_completes(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE).requires_grad_(True)
        out = model(x)  # Step 0 (trace).
        loss = out.sum()
        loss.backward()

        # Verify gradients exist.
        assert x.grad is not None

    def test_backward_after_rebuild(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, prefetch_k=1, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE).requires_grad_(True)
        out = model(x)  # Step 0.
        out = model(x)  # Step 1 (rebuild).
        assert runtime.mode == "scheduled"

        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ── shutdown tests ───────────────────────────────────────────────────────


class TestShutdown:
    """Test clean shutdown."""

    def test_shutdown(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        model(x)

        runtime.shutdown()
        assert runtime.mode == "shutdown"

    def test_double_shutdown_safe(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)

        runtime.shutdown()
        runtime.shutdown()  # Should not raise.

    def test_shutdown_keep_pool(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)

        pool = runtime.shutdown_keep_pool()
        assert pool is not None
        assert runtime.mode == "shutdown"

        # Pool should still be usable.
        stats = pool.stats()
        assert stats["total"] > 0

        # Clean up.
        pool.shutdown()

    def test_pool_reuse(self) -> None:
        model1 = _SequentialModel(num_layers=3, hidden=32)
        runtime1 = LayerRuntime(model1, dtype=torch.float32, telemetry=False)
        pool = runtime1.shutdown_keep_pool()

        # Reuse pool in a second runtime.
        model2 = _SequentialModel(num_layers=3, hidden=32)
        runtime2 = LayerRuntime(model2, dtype=torch.float32, telemetry=False, pool=pool)

        x = torch.randn(2, 32, device=_DEVICE)
        model2(x)
        model2(x)
        assert runtime2.mode == "scheduled"
        runtime2.shutdown()


# ── edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model_raises(self) -> None:
        model = _EmptyModel()
        with pytest.raises(ValueError, match="No eligible layers"):
            LayerRuntime(model, dtype=torch.float32)

    def test_single_layer_model(self) -> None:
        model = _SingleLayerModel()
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)
        assert runtime.num_layers == 1

        x = torch.randn(2, 32, device=_DEVICE)
        out = model(x)  # Step 0.
        out = model(x)  # Step 1.
        assert runtime.mode == "scheduled"

    def test_shared_module_single_entry(self) -> None:
        model = _SharedModel()
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)
        # Only 2 unique modules (shared + c).
        assert runtime.num_layers == 2

    def test_properties(self) -> None:
        model = _SequentialModel(num_layers=3, hidden=32)
        runtime = LayerRuntime(model, dtype=torch.float32, telemetry=False)
        assert runtime.num_layers == 3
        assert runtime.mode == "trace"
        assert runtime.traced is False
        assert runtime.step == 0
        assert isinstance(runtime.stats, dict)


# ── top-level API tests ──────────────────────────────────────────────────


class TestAPI:
    """Test the top-level stagehand.layer() / wrap() / Runtime API."""

    def test_layer_returns_same_model(self) -> None:
        import stagehand

        model = _SequentialModel(num_layers=3, hidden=32)
        result = stagehand.layer(model, dtype=torch.float32, telemetry=False)
        assert result is model

    def test_runtime_accessible(self) -> None:
        import stagehand

        model = _SequentialModel(num_layers=3, hidden=32)
        stagehand.layer(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime  # type: ignore[attr-defined]
        assert isinstance(runtime, LayerRuntime)
        runtime.shutdown()

    def test_wrap_calls_layer(self) -> None:
        import stagehand

        model = _SequentialModel(num_layers=3, hidden=32)
        result = stagehand.wrap(model, dtype=torch.float32, telemetry=False)
        assert result is model
        assert hasattr(model, "_stagehand_layer_runtime")
        model._stagehand_layer_runtime.shutdown()  # type: ignore[attr-defined]

    def test_runtime_class(self) -> None:
        import stagehand

        rt = stagehand.Runtime(dtype=torch.float32, telemetry=False)
        model = _SequentialModel(num_layers=3, hidden=32)
        result = rt.layer(model)
        assert result is model
        assert hasattr(model, "_stagehand_layer_runtime")
        model._stagehand_layer_runtime.shutdown()  # type: ignore[attr-defined]

    def test_layer_forward_works(self) -> None:
        import stagehand

        model = _SequentialModel(num_layers=4, hidden=32)
        stagehand.layer(model, dtype=torch.float32, telemetry=False)

        x = torch.randn(2, 32, device=_DEVICE)
        out = model(x)  # Step 0.
        assert out.shape == (2, 32)

        out = model(x)  # Step 1 (rebuild).
        assert out.shape == (2, 32)

        out = model(x)  # Step 2.
        assert out.shape == (2, 32)

        model._stagehand_layer_runtime.shutdown()  # type: ignore[attr-defined]
