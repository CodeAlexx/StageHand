"""AdamW optimizer that keeps all state on CPU for use with stagehand.layer().

Standard PyTorch optimizers create state tensors (momentum, variance) on
whatever device the param is on at the first ``step()`` call.  With Stagehand
layer-mode offloading, params live on CPU after ``end_step()``, so states
created here naturally stay on CPU — no extra GPU VRAM for optimizer state.

Steps params one at a time using ``torch.optim._functional.adamw`` — the same
kernel PyTorch uses internally.  CPU Adam is O(n) element-wise ops (~1ms per
2MB weight), which is negligible compared to forward/backward with offloading.
"""
from __future__ import annotations

import torch
from torch.optim._functional import adamw as _functional_adamw

__all__ = ["OffloadedAdamW"]


class OffloadedAdamW(torch.optim.Optimizer):
    """AdamW that keeps optimizer states on the same device as each param.

    With ``stagehand.layer()`` offloading, params live on CPU after
    ``end_step()``, so states are created on CPU and stay there.

    Uses ``torch.optim._functional.adamw`` with single-param lists to
    step params one at a time — no batching across params, no device
    transfers.

    Parameters
    ----------
    params:
        Iterable of parameters or param groups.
    lr:
        Learning rate.
    betas:
        Coefficients for computing running averages of gradient and its square.
    eps:
        Term added to denominator for numerical stability.
    weight_decay:
        Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Steps each parameter individually using the functional AdamW kernel.
        States are lazily initialized on the param's current device.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.device != p.device:
                    grad = grad.to(p.device)

                state = self.state[p]

                # Lazy state initialization on param's device.
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                _functional_adamw(
                    [p],
                    [grad],
                    [state["exp_avg"]],
                    [state["exp_avg_sq"]],
                    [],                  # max_exp_avg_sqs (amsgrad=False)
                    [state["step"]],
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                    foreach=False,
                    capturable=False,
                    differentiable=False,
                    has_complex=False,
                )

        return loss
