"""Tests for OffloadedAdamW — CPU-resident optimizer for stagehand offloading.

All tests run CPU-only.  CUDA-gated tests use ``@pytest.mark.skipif``.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from stagehand.optim import OffloadedAdamW

__all__: list[str] = []

_HAS_CUDA = torch.cuda.is_available()


# ── helpers ──────────────────────────────────────────────────────────────


class _SimpleModel(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _quadratic_loss(params: list[torch.Tensor]) -> torch.Tensor:
    """Sum of squared params — minimum at zero."""
    return sum(p.pow(2).sum() for p in params)


# ── basic correctness ────────────────────────────────────────────────────


class TestBasicCorrectness:
    """OffloadedAdamW produces the same results as torch.optim.AdamW."""

    def test_single_step_parity(self) -> None:
        """One step matches torch.optim.AdamW to floating point precision."""
        torch.manual_seed(42)
        p_ours = nn.Parameter(torch.randn(64, 64))
        p_ref = nn.Parameter(p_ours.data.clone())

        # Same grad for both.
        grad = torch.randn_like(p_ours)
        p_ours.grad = grad.clone()
        p_ref.grad = grad.clone()

        lr, betas, eps, wd = 1e-3, (0.9, 0.999), 1e-8, 0.01

        opt_ours = OffloadedAdamW([p_ours], lr=lr, betas=betas, eps=eps, weight_decay=wd)
        opt_ref = torch.optim.AdamW([p_ref], lr=lr, betas=betas, eps=eps, weight_decay=wd)

        opt_ours.step()
        opt_ref.step()

        torch.testing.assert_close(p_ours.data, p_ref.data, atol=1e-6, rtol=1e-5)

    def test_multi_step_parity(self) -> None:
        """Multiple steps match torch.optim.AdamW."""
        torch.manual_seed(123)
        p_ours = nn.Parameter(torch.randn(32, 32))
        p_ref = nn.Parameter(p_ours.data.clone())

        lr, betas, eps, wd = 1e-3, (0.9, 0.999), 1e-8, 0.01
        opt_ours = OffloadedAdamW([p_ours], lr=lr, betas=betas, eps=eps, weight_decay=wd)
        opt_ref = torch.optim.AdamW([p_ref], lr=lr, betas=betas, eps=eps, weight_decay=wd)

        for _ in range(10):
            grad = torch.randn_like(p_ours)
            p_ours.grad = grad.clone()
            p_ref.grad = grad.clone()
            opt_ours.step()
            opt_ref.step()

        torch.testing.assert_close(p_ours.data, p_ref.data, atol=1e-5, rtol=1e-4)

    def test_convergence_on_quadratic(self) -> None:
        """Loss decreases on a simple quadratic objective."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(100))
        opt = OffloadedAdamW([p], lr=0.01, weight_decay=0.0)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = p.pow(2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, f"Loss didn't converge: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_weight_decay_shrinks_params(self) -> None:
        """With weight decay and zero grad, params shrink toward zero."""
        torch.manual_seed(7)
        p = nn.Parameter(torch.ones(32))
        opt = OffloadedAdamW([p], lr=0.01, weight_decay=0.1)

        initial_norm = p.data.norm().item()
        for _ in range(100):
            p.grad = torch.zeros_like(p)
            opt.step()

        assert p.data.norm().item() < initial_norm * 0.95

    def test_zero_grad_no_update(self) -> None:
        """Param with no grad is not updated."""
        p_with = nn.Parameter(torch.ones(16))
        p_without = nn.Parameter(torch.ones(16))
        opt = OffloadedAdamW([p_with, p_without], lr=0.01)

        p_with.grad = torch.ones_like(p_with)
        # p_without.grad remains None

        original = p_without.data.clone()
        opt.step()

        torch.testing.assert_close(p_without.data, original)

    def test_states_on_param_device(self) -> None:
        """Optimizer states are created on the same device as the param (CPU)."""
        p = nn.Parameter(torch.randn(32))
        opt = OffloadedAdamW([p], lr=0.01)
        p.grad = torch.randn_like(p)
        opt.step()

        state = opt.state[p]
        assert state["exp_avg"].device == p.device
        assert state["exp_avg_sq"].device == p.device
        assert state["step"].device == p.device


# ── per-param stepping ───────────────────────────────────────────────────


class TestPerParamStepping:
    """States are independent per param, mixed dtypes work."""

    def test_independent_states(self) -> None:
        """Different grads produce different state values."""
        p1 = nn.Parameter(torch.ones(16))
        p2 = nn.Parameter(torch.ones(16))
        opt = OffloadedAdamW([p1, p2], lr=0.01)

        p1.grad = torch.ones_like(p1) * 10.0
        p2.grad = torch.ones_like(p2) * 0.001
        opt.step()

        # States should differ because grads differ.
        assert not torch.allclose(opt.state[p1]["exp_avg"], opt.state[p2]["exp_avg"])

    def test_mixed_dtypes(self) -> None:
        """Float32 and bfloat16 params in one group both get stepped."""
        p_f32 = nn.Parameter(torch.randn(16, dtype=torch.float32))
        p_bf16 = nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
        opt = OffloadedAdamW([p_f32, p_bf16], lr=0.01)

        p_f32.grad = torch.randn_like(p_f32)
        p_bf16.grad = torch.randn_like(p_bf16)

        orig_f32 = p_f32.data.clone()
        orig_bf16 = p_bf16.data.clone()
        opt.step()

        assert not torch.equal(p_f32.data, orig_f32)
        assert not torch.equal(p_bf16.data, orig_bf16)

    def test_no_grad_param_skipped(self) -> None:
        """Params without grad are skipped, no state created."""
        p1 = nn.Parameter(torch.randn(16))
        p2 = nn.Parameter(torch.randn(16))
        opt = OffloadedAdamW([p1, p2], lr=0.01)

        p1.grad = torch.randn_like(p1)
        # p2 has no grad
        opt.step()

        assert p1 in opt.state
        assert p2 not in opt.state


# ── stagehand integration ────────────────────────────────────────────────


class TestStagehandIntegration:
    """Optimizer works with stagehand-like CPU param/grad patterns."""

    def test_cpu_param_cpu_grad(self) -> None:
        """Standard stagehand post-end_step scenario: param and grad on CPU."""
        model = _SimpleModel(hidden=32)
        model.cpu()
        opt = OffloadedAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, 32)
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()

        # All states on CPU.
        for p in model.parameters():
            if p in opt.state:
                assert opt.state[p]["exp_avg"].device.type == "cpu"
                assert opt.state[p]["exp_avg_sq"].device.type == "cpu"

    def test_states_stay_cpu_after_multiple_steps(self) -> None:
        """States remain on CPU after multiple train steps."""
        model = _SimpleModel(hidden=16)
        model.cpu()
        opt = OffloadedAdamW(model.parameters(), lr=1e-3)

        for _ in range(5):
            x = torch.randn(2, 16)
            opt.zero_grad()
            loss = model(x).pow(2).sum()
            loss.backward()
            opt.step()

        for p in model.parameters():
            if p in opt.state:
                assert opt.state[p]["exp_avg"].device.type == "cpu"
                assert opt.state[p]["exp_avg_sq"].device.type == "cpu"

    def test_param_updated_inplace_on_cpu(self) -> None:
        """param.data is updated in-place — stagehand scheduler can read it."""
        p = nn.Parameter(torch.ones(32, device="cpu"))
        opt = OffloadedAdamW([p], lr=0.1, weight_decay=0.0)

        data_ptr_before = p.data.data_ptr()
        p.grad = torch.ones_like(p)
        opt.step()
        data_ptr_after = p.data.data_ptr()

        # In-place update: same storage.
        assert data_ptr_before == data_ptr_after
        # But values changed.
        assert not torch.allclose(p.data, torch.ones(32))

    def test_full_train_cycle_loss_decreases(self) -> None:
        """Forward → backward → step cycle reduces loss over time."""
        torch.manual_seed(99)
        model = _SimpleModel(hidden=16)
        model.cpu()
        opt = OffloadedAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

        losses = []
        for _ in range(30):
            x = torch.randn(8, 16)
            target = torch.zeros(8, 16)
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease (compare first 5 avg vs last 5 avg).
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} → {late_avg:.4f}"


# ── state dict save/load ─────────────────────────────────────────────────


class TestStateDict:
    """state_dict() / load_state_dict() roundtrip."""

    def test_roundtrip_preserves_states(self) -> None:
        """Save and load preserves exp_avg, exp_avg_sq values."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32))
        opt = OffloadedAdamW([p], lr=0.01)
        p.grad = torch.randn_like(p)
        opt.step()
        opt.step()

        sd = copy.deepcopy(opt.state_dict())

        # Create fresh optimizer and load state.
        opt2 = OffloadedAdamW([p], lr=0.01)
        opt2.load_state_dict(sd)

        # Compare states.
        s1 = opt.state[p]
        s2 = opt2.state[p]
        torch.testing.assert_close(s1["exp_avg"], s2["exp_avg"])
        torch.testing.assert_close(s1["exp_avg_sq"], s2["exp_avg_sq"])

    def test_states_cpu_after_load(self) -> None:
        """States stay on CPU after load_state_dict."""
        p = nn.Parameter(torch.randn(16))
        opt = OffloadedAdamW([p], lr=0.01)
        p.grad = torch.randn_like(p)
        opt.step()

        sd = opt.state_dict()
        opt2 = OffloadedAdamW([p], lr=0.01)
        opt2.load_state_dict(sd)

        state = opt2.state[p]
        assert state["exp_avg"].device.type == "cpu"
        assert state["exp_avg_sq"].device.type == "cpu"

    def test_step_count_preserved(self) -> None:
        """Step counter is preserved across save/load."""
        p = nn.Parameter(torch.randn(16))
        opt = OffloadedAdamW([p], lr=0.01)

        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()

        sd = opt.state_dict()
        opt2 = OffloadedAdamW([p], lr=0.01)
        opt2.load_state_dict(sd)

        state = opt2.state[p]
        assert state["step"].item() == 5.0


# ── edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty groups, cross-device grads, closure support."""

    def test_empty_param_group(self) -> None:
        """Empty param list raises ValueError (PyTorch requirement)."""
        with pytest.raises(ValueError, match="empty parameter list"):
            OffloadedAdamW([], lr=0.01)

    def test_large_param_stepping(self) -> None:
        """Large params step correctly (regression for vectorization issues)."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(1024, 1024))
        opt = OffloadedAdamW([p], lr=1e-3)

        p.grad = torch.randn_like(p)
        original = p.data.clone()
        opt.step()

        assert not torch.equal(p.data, original)
        assert p.data.isfinite().all()

    def test_closure_support(self) -> None:
        """Closure is called and loss returned."""
        p = nn.Parameter(torch.randn(16))
        opt = OffloadedAdamW([p], lr=0.01)

        call_count = 0

        def closure():
            nonlocal call_count
            call_count += 1
            opt.zero_grad()
            loss = p.pow(2).sum()
            loss.backward()
            return loss

        loss = opt.step(closure=closure)
        assert call_count == 1
        assert loss is not None
        assert loss.item() > 0

    def test_multiple_param_groups(self) -> None:
        """Different param groups with different lr/wd work."""
        p1 = nn.Parameter(torch.randn(16))
        p2 = nn.Parameter(torch.randn(16))

        opt = OffloadedAdamW([
            {"params": [p1], "lr": 0.1, "weight_decay": 0.0},
            {"params": [p2], "lr": 0.001, "weight_decay": 0.1},
        ])

        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)

        orig1 = p1.data.clone()
        orig2 = p2.data.clone()
        opt.step()

        # Both updated.
        assert not torch.equal(p1.data, orig1)
        assert not torch.equal(p2.data, orig2)

        # p1 should have a larger update (higher lr).
        delta1 = (p1.data - orig1).abs().mean()
        delta2 = (p2.data - orig2).abs().mean()
        assert delta1 > delta2


# ── stagehand.layer() end-to-end ─────────────────────────────────────────


class TestWithStagehandLayer:
    """Integration with stagehand.layer() — the real use case."""

    def test_layer_mode_train_cycle(self) -> None:
        """stagehand.layer() + OffloadedAdamW: full train loop works."""
        import stagehand

        device = "cuda" if _HAS_CUDA else "cpu"
        model = _SimpleModel(hidden=16)
        model = stagehand.layer(model, dtype=torch.float32, inference_mode=False)
        opt = OffloadedAdamW(model.parameters(), lr=1e-3)

        losses = []
        for step in range(10):
            x = torch.randn(4, 16, device=device)
            opt.zero_grad()
            out = model(x)
            loss = out.pow(2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model._stagehand_layer_runtime.shutdown()  # type: ignore[attr-defined]

        # Should not crash and loss should be finite.
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_layer_mode_states_match_param_device(self) -> None:
        """Optimizer states are created on the same device as params at step time.

        With stagehand layer mode, params may be on GPU during backward.
        States are lazily created on whatever device the param is on when
        step() is first called.  In real training with block-mode's explicit
        end_step(), params are on CPU — so states stay on CPU.
        """
        import stagehand

        device = "cuda" if _HAS_CUDA else "cpu"
        model = _SimpleModel(hidden=16)
        model = stagehand.layer(model, dtype=torch.float32, inference_mode=False)
        opt = OffloadedAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, 16, device=device)
        opt.zero_grad()
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()

        model._stagehand_layer_runtime.shutdown()  # type: ignore[attr-defined]

        # States should be on the same device as the param was at step() time.
        for p in model.parameters():
            if p in opt.state:
                for key in ("exp_avg", "exp_avg_sq"):
                    assert opt.state[p][key].device.type == p.device.type, (
                        f"State '{key}' device mismatch"
                    )
