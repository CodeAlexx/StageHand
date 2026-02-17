"""Tests for the RamTorch compatibility shim.

Verifies that stagehand.compat.ramtorch provides a drop-in replacement
for ramtorch.helpers, setting is_ramtorch flags and forwarding kwargs
to stagehand.layer().
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from stagehand.compat.ramtorch import (
    Linear,
    _mark_is_ramtorch,
    move_model_to_device,
    reattach_is_ramtorch_flags,
    replace_linear_with_ramtorch,
)
from stagehand.layer import LAYER_TYPES, LayerRuntime

__all__: list[str] = []

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── test models ──────────────────────────────────────────────────────────


class _SimpleMLP(nn.Module):
    """Simple MLP with Linear layers."""

    def __init__(self, hidden: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _MixedModel(nn.Module):
    """Model with Linear, Conv2d, LayerNorm (non-managed), and Embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.norm = nn.LayerNorm(32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ConvModel(nn.Module):
    """Model with Conv2d + Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _EmptyModel(nn.Module):
    """Model with no eligible layers."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class _LayerNormOnly(nn.Module):
    """Model with only LayerNorm (no eligible layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.norm1(x))


class _EmbeddingLinearModel(nn.Module):
    """Model with Embedding + Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, 32)
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── 1. API Compat ────────────────────────────────────────────────────────


class TestAPICompat:
    """Verify the compat shim matches ramtorch's API surface."""

    def test_replace_returns_same_instance(self) -> None:
        model = _SimpleMLP()
        result = replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        assert result is model
        model._stagehand_layer_runtime.shutdown()

    def test_model_has_runtime_attribute(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        assert hasattr(model, "_stagehand_layer_runtime")
        assert isinstance(model._stagehand_layer_runtime, LayerRuntime)
        model._stagehand_layer_runtime.shutdown()

    def test_managed_linear_modules_have_is_ramtorch(self) -> None:
        model = _SimpleMLP(num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            for _name, mod in runtime._layer_map.items():
                assert getattr(mod, "is_ramtorch", False) is True
        finally:
            runtime.shutdown()

    def test_managed_params_have_is_ramtorch(self) -> None:
        model = _SimpleMLP(num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    assert getattr(p, "is_ramtorch", False) is True
        finally:
            runtime.shutdown()

    def test_non_managed_modules_no_is_ramtorch(self) -> None:
        model = _MixedModel()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            managed_ids = {id(m) for m in runtime._layer_map.values()}
            for module in model.modules():
                if module is model:
                    continue
                if id(module) not in managed_ids:
                    assert not getattr(module, "is_ramtorch", False), (
                        f"Non-managed module {type(module).__name__} should not have is_ramtorch"
                    )
        finally:
            runtime.shutdown()

    def test_move_model_to_device_noop_after_replace(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        result = move_model_to_device(model, "cuda")
        assert result is model
        model._stagehand_layer_runtime.shutdown()

    def test_reattach_restores_flags_after_clear(self) -> None:
        model = _SimpleMLP(num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            # Clear flags on all params.
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    if hasattr(p, "is_ramtorch"):
                        del p.is_ramtorch

            # Verify cleared.
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    assert not getattr(p, "is_ramtorch", False)

            # Reattach.
            reattach_is_ramtorch_flags(model)

            # Verify restored.
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    assert getattr(p, "is_ramtorch", False) is True
        finally:
            runtime.shutdown()

    def test_linear_stub_has_is_ramtorch(self) -> None:
        assert Linear.is_ramtorch is True
        stub = Linear(10, 5)
        assert getattr(stub, "is_ramtorch", False) is True
        assert isinstance(stub, nn.Linear)


# ── 2. Kwargs Passthrough ────────────────────────────────────────────────


class TestKwargsPassthrough:
    """Verify stagehand.layer() kwargs are forwarded correctly."""

    def test_prefetch_k_forwarded(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(
            model, dtype=torch.float32, telemetry=False, prefetch_k=5,
        )
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime._prefetch_k == 5
        finally:
            runtime.shutdown()

    def test_inference_mode_forwarded(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(
            model, dtype=torch.float32, telemetry=False, inference_mode=True,
        )
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime._inference_mode is True
        finally:
            runtime.shutdown()

    def test_dtype_forwarded(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(
            model, dtype=torch.float32, telemetry=False,
        )
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime._dtype == torch.float32
        finally:
            runtime.shutdown()

    def test_default_kwargs_produce_working_runtime(self) -> None:
        model = _SimpleMLP()
        # Use float32 for CPU compat, but otherwise defaults.
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime.mode == "trace"
            assert runtime.num_layers == 4
        finally:
            runtime.shutdown()


# ── 3. Functional Correctness ────────────────────────────────────────────


class TestFunctionalCorrectness:
    """Verify the wrapped model computes correctly."""

    def test_simple_mlp_forward(self) -> None:
        model = _SimpleMLP(hidden=16, num_layers=3)

        # Reference output before wrapping (on CPU).
        x_cpu = torch.randn(2, 16)
        with torch.no_grad():
            ref = model(x_cpu).clone()

        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            # Input must be on same device as model params.
            x = x_cpu.to(_DEVICE)

            # Step 0: trace pass.
            with torch.no_grad():
                out0 = model(x)
            assert out0.shape == ref.shape
            assert torch.allclose(out0.cpu(), ref, atol=1e-5), "Trace pass output mismatch"

            # Step 1: scheduled pass.
            with torch.no_grad():
                out1 = model(x)
            assert torch.allclose(out1.cpu(), ref, atol=1e-5), "Scheduled pass output mismatch"
        finally:
            runtime.shutdown()

    def test_multi_step_inference(self) -> None:
        model = _SimpleMLP(hidden=16, num_layers=3)
        x_cpu = torch.randn(2, 16)

        with torch.no_grad():
            ref = model(x_cpu).clone()

        replace_linear_with_ramtorch(
            model, dtype=torch.float32, telemetry=False, inference_mode=True,
        )
        runtime = model._stagehand_layer_runtime
        try:
            x = x_cpu.to(_DEVICE)
            # Run 5 steps.
            for i in range(5):
                with torch.no_grad():
                    out = model(x)
                assert torch.allclose(out.cpu(), ref, atol=1e-5), f"Step {i} output mismatch"
        finally:
            runtime.shutdown()

    def test_backward_pass_gradients(self) -> None:
        model = _SimpleMLP(hidden=16, num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            x = torch.randn(2, 16, device=_DEVICE)

            # Step 0 (trace): forward + backward.
            out0 = model(x)
            out0.sum().backward()

            # Verify gradients exist on managed params.
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    assert p.grad is not None, f"Missing grad on param in {_name}"

            # Zero grads.
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Step 1 (scheduled): forward + backward.
            out1 = model(x)
            out1.sum().backward()

            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    assert p.grad is not None, f"Missing grad on param in {_name} (step 1)"
        finally:
            runtime.shutdown()

    def test_conv_and_linear_both_managed(self) -> None:
        model = _ConvModel()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime.num_layers == 2
            layer_names = list(runtime._layer_map.keys())
            assert "conv" in layer_names
            assert "fc" in layer_names
        finally:
            runtime.shutdown()

    def test_embedding_and_linear_both_managed(self) -> None:
        model = _EmbeddingLinearModel()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime.num_layers == 2
            layer_names = list(runtime._layer_map.keys())
            assert "embed" in layer_names
            assert "fc" in layer_names
        finally:
            runtime.shutdown()

    def test_inference_mode_no_backward_hooks(self) -> None:
        model = _SimpleMLP(hidden=16, num_layers=2)
        replace_linear_with_ramtorch(
            model, dtype=torch.float32, telemetry=False, inference_mode=True,
        )
        runtime = model._stagehand_layer_runtime
        try:
            assert runtime._inference_mode is True
            # In inference mode, only forward hooks are installed (2 per layer).
            # In training mode, 4 per layer (forward pre+post, backward pre+post).
            hooks_per_layer = len(runtime._hook_handles) / runtime.num_layers
            assert hooks_per_layer == 2, (
                f"Expected 2 hooks/layer in inference mode, got {hooks_per_layer}"
            )
        finally:
            runtime.shutdown()


# ── 4. SimpleTuner Compat Patterns ───────────────────────────────────────


class TestSimpleTunerCompat:
    """Verify patterns used by SimpleTuner for RamTorch integration."""

    def test_quantization_skip_pattern(self) -> None:
        """SimpleTuner: any(getattr(p, 'is_ramtorch', False) for p in model.parameters())"""
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        try:
            has_ramtorch = any(
                getattr(p, "is_ramtorch", False) for p in model.parameters()
            )
            assert has_ramtorch is True
        finally:
            model._stagehand_layer_runtime.shutdown()

    def test_ddp_ignore_pattern(self) -> None:
        """SimpleTuner: collect param names with is_ramtorch for DDP ignore list."""
        model = _SimpleMLP(num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        try:
            ramtorch_params = [
                name for name, p in model.named_parameters()
                if getattr(p, "is_ramtorch", False)
            ]
            # All Linear params should be flagged.
            assert len(ramtorch_params) == 3  # 3 layers, 1 weight each (no bias)
        finally:
            model._stagehand_layer_runtime.shutdown()

    def test_device_move_skip_is_safe(self) -> None:
        """SimpleTuner: move_model_to_device after replace is safe no-op."""
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        try:
            # Should not raise or change anything.
            result = move_model_to_device(model, "cuda")
            assert result is model
            # Runtime still active.
            assert model._stagehand_layer_runtime.mode in ("trace", "scheduled")
        finally:
            model._stagehand_layer_runtime.shutdown()

    def test_mixed_managed_unmanaged_params(self) -> None:
        """Managed params have is_ramtorch, unmanaged params don't."""
        model = _MixedModel()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            managed_param_ids = set()
            for _name, mod in runtime._layer_map.items():
                for p in mod.parameters(recurse=False):
                    managed_param_ids.add(id(p))

            for name, p in model.named_parameters():
                if id(p) in managed_param_ids:
                    assert getattr(p, "is_ramtorch", False) is True, (
                        f"Managed param {name} missing is_ramtorch"
                    )
                else:
                    assert not getattr(p, "is_ramtorch", False), (
                        f"Unmanaged param {name} should not have is_ramtorch"
                    )
        finally:
            runtime.shutdown()

    def test_standard_optimizer_works(self) -> None:
        """Standard PyTorch optimizer works with model after replace."""
        model = _SimpleMLP(hidden=16, num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.randn(2, 16, device=_DEVICE)

            # Step 0 (trace).
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step 1 (scheduled).
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        finally:
            runtime.shutdown()

    def test_module_is_ramtorch_check(self) -> None:
        """SimpleTuner: getattr(module, 'is_ramtorch', False) on managed modules."""
        model = _SimpleMLP(num_layers=3)
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        try:
            for _name, mod in runtime._layer_map.items():
                assert getattr(mod, "is_ramtorch", False) is True
        finally:
            runtime.shutdown()


# ── 5. Shutdown & Cleanup ────────────────────────────────────────────────


class TestShutdownCleanup:
    """Verify clean shutdown behavior."""

    def test_shutdown_after_replace(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        runtime.shutdown()
        assert runtime.mode == "shutdown"

    def test_pool_reuse_via_shutdown_keep_pool(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        pool = runtime.shutdown_keep_pool()
        assert pool is not None
        assert runtime.mode == "shutdown"

        # Pool can be reused for another model.
        model2 = _SimpleMLP()
        replace_linear_with_ramtorch(
            model2, dtype=torch.float32, telemetry=False, pool=pool,
        )
        runtime2 = model2._stagehand_layer_runtime
        assert runtime2._pool is pool
        runtime2.shutdown()

    def test_double_shutdown_is_safe(self) -> None:
        model = _SimpleMLP()
        replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)
        runtime = model._stagehand_layer_runtime
        runtime.shutdown()
        runtime.shutdown()  # Should not raise.
        assert runtime.mode == "shutdown"


# ── 6. Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_model_raises(self) -> None:
        model = _EmptyModel()
        with pytest.raises(ValueError, match="No eligible layers"):
            replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)

    def test_layernorm_only_raises(self) -> None:
        model = _LayerNormOnly()
        with pytest.raises(ValueError, match="No eligible layers"):
            replace_linear_with_ramtorch(model, dtype=torch.float32, telemetry=False)

    def test_move_without_prior_replace_does_standard_move(self) -> None:
        model = _SimpleMLP(hidden=8, num_layers=2)
        # Model not wrapped — move_model_to_device should do standard move.
        assert not hasattr(model, "_stagehand_layer_runtime")
        result = move_model_to_device(model, "cpu")
        assert result is model
        for p in model.parameters():
            assert p.device == torch.device("cpu")

    def test_reattach_on_non_stagehand_model_is_noop(self) -> None:
        model = _SimpleMLP()
        # Should not raise.
        reattach_is_ramtorch_flags(model)
        # No flags set since model isn't wrapped.
        for p in model.parameters():
            assert not getattr(p, "is_ramtorch", False)

    def test_linear_stub_isinstance(self) -> None:
        """Linear stub works for isinstance checks."""
        stub = Linear(10, 5)
        assert isinstance(stub, nn.Linear)
        assert isinstance(stub, Linear)
        assert stub.is_ramtorch is True

    def test_replace_via_compat_init(self) -> None:
        """Import via stagehand.compat works."""
        from stagehand.compat import replace_linear_with_ramtorch as replace_fn

        model = _SimpleMLP()
        replace_fn(model, dtype=torch.float32, telemetry=False)
        try:
            assert hasattr(model, "_stagehand_layer_runtime")
        finally:
            model._stagehand_layer_runtime.shutdown()
