"""RamTorch compatibility shim — drop-in replacement using stagehand.layer().

Provides the same public API as ``ramtorch.helpers`` so that codebases
using RamTorch (e.g. SimpleTuner) can switch to Stagehand with zero
code changes::

    # Before:
    from ramtorch.helpers import replace_linear_with_ramtorch, move_model_to_device

    # After:
    from stagehand.compat.ramtorch import replace_linear_with_ramtorch, move_model_to_device

The shim wraps the model with :func:`stagehand.layer` internally and
sets ``is_ramtorch = True`` flags on managed modules/params so that
downstream checks (quantization skip, DDP ignore, device move skip)
continue to work.
"""
from __future__ import annotations

import torch
from torch import nn

from stagehand.layer import LayerRuntime

__all__ = [
    "replace_linear_with_ramtorch",
    "move_model_to_device",
    "reattach_is_ramtorch_flags",
    "Linear",
]


def replace_linear_with_ramtorch(
    module: nn.Module,
    device: str = "cuda",
    *,
    vram_budget: str | int | None = None,
    ram_budget: str | int | None = None,
    prefetch_k: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    inference_mode: bool = False,
    telemetry: bool = True,
    pool: object | None = None,
) -> nn.Module:
    """Drop-in replacement for ``ramtorch.helpers.replace_linear_with_ramtorch``.

    Internally wraps *module* with :func:`stagehand.layer` and marks managed
    modules/params with ``is_ramtorch = True`` for downstream compat checks.

    Parameters
    ----------
    module:
        The model to wrap.
    device:
        Target GPU device (ignored — stagehand auto-detects).
    vram_budget, ram_budget, prefetch_k, dtype, inference_mode, telemetry, pool:
        Forwarded to :func:`stagehand.layer`.

    Returns
    -------
    The same model instance, now managed by stagehand with ``is_ramtorch``
    flags set on all managed modules and parameters.
    """
    import stagehand

    kwargs: dict = dict(
        vram_budget=vram_budget,
        ram_budget=ram_budget,
        prefetch_k=prefetch_k,
        dtype=dtype,
        inference_mode=inference_mode,
        telemetry=telemetry,
    )
    if pool is not None:
        kwargs["pool"] = pool

    model = stagehand.layer(module, **kwargs)
    runtime: LayerRuntime = model._stagehand_layer_runtime  # type: ignore[attr-defined]
    _mark_is_ramtorch(runtime)
    return model


def move_model_to_device(
    model: nn.Module,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Drop-in replacement for ``ramtorch.helpers.move_model_to_device``.

    No-op if stagehand is active (LayerRuntime already moved non-managed
    params to GPU during init). Otherwise falls back to a standard
    parameter/buffer device move.
    """
    if hasattr(model, "_stagehand_layer_runtime"):
        return model

    # Fallback: standard move for non-stagehand models.
    target = torch.device(device)
    for p in model.parameters():
        if p.device != target:
            p.data = p.data.to(target)
    for name, buf in model.named_buffers():
        if buf.device != target:
            # Walk to parent module to set buffer properly.
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], buf.to(target))
            else:
                setattr(model, name, buf.to(target))
    return model


def reattach_is_ramtorch_flags(module: nn.Module) -> None:
    """Drop-in replacement for ``ramtorch.helpers.reattach_is_ramtorch_flags``.

    Re-walks managed modules and re-sets ``is_ramtorch`` on params.
    Useful after deserialization (``torch.save``/``torch.load`` cycle)
    which strips custom attributes from tensors.
    """
    if hasattr(module, "_stagehand_layer_runtime"):
        runtime: LayerRuntime = module._stagehand_layer_runtime  # type: ignore[attr-defined]
        _mark_is_ramtorch(runtime)


def _mark_is_ramtorch(runtime: LayerRuntime) -> None:
    """Set ``is_ramtorch = True`` on all managed modules and their params."""
    for _name, mod in runtime._layer_map.items():
        mod.is_ramtorch = True  # type: ignore[attr-defined]
        for p in mod.parameters(recurse=False):
            p.is_ramtorch = True  # type: ignore[attr-defined]


class Linear(nn.Linear):
    """Stub for ``isinstance`` checks against ``ramtorch.Linear``.

    This class is NOT used for computation — stagehand manages the
    original ``nn.Linear`` modules via hooks. It exists solely so that
    ``isinstance(module, ramtorch.Linear)`` checks work in downstream code.
    """

    is_ramtorch: bool = True
