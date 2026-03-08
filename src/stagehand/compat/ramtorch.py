"""RamTorch compatibility shim — drop-in replacement using stagehand.layer().

Provides the same public API as ``ramtorch.helpers`` so that codebases
using RamTorch can switch to Stagehand with a one-line import change::

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

from collections import OrderedDict
from typing import Callable, Sequence

import torch
from torch import nn

from stagehand.layer import LayerRuntime

__all__ = [
    "replace_linear_with_ramtorch",
    "move_model_to_device",
    "reattach_is_ramtorch_flags",
    "add_custom_hooks",
    "register_ramtorch_hook",
    "register_ramtorch_grad_hook",
    "register_ramtorch_post_accumulate_grad_hook",
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
    params to GPU during init). Otherwise falls back to the same logic as
    real RamTorch: move params/buffers to *device*, skipping any that have
    ``is_ramtorch = True``.
    """
    if hasattr(model, "_stagehand_layer_runtime"):
        return model

    # Fallback: match real RamTorch behavior — skip is_ramtorch params/buffers.
    target = torch.device(device)

    for _name, param in model.named_parameters(recurse=True):
        if getattr(param, "is_ramtorch", False):
            continue
        if param.device != target:
            with torch.no_grad():
                new_param = param.to(target)
            param.data = new_param
            if param._grad is not None:
                param._grad = param._grad.to(target)

    for full_name, buf in model.named_buffers(recurse=True):
        if getattr(buf, "is_ramtorch", False):
            continue
        if buf.device == target:
            continue
        with torch.no_grad():
            new_buf = buf.to(target)
        # Walk to the owning module to set buffer properly.
        parent = model
        *parents, attr = full_name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        parent._buffers[attr] = new_buf

    return model


def reattach_is_ramtorch_flags(module: nn.Module) -> None:
    """Drop-in replacement for ``ramtorch.helpers.reattach_is_ramtorch_flags``.

    If stagehand is active, re-walks managed modules and re-sets
    ``is_ramtorch`` on params and buffers. Otherwise walks all modules
    recursively (matching real RamTorch behavior): any module with
    ``is_ramtorch = True`` gets its params and buffers flagged.
    """
    if hasattr(module, "_stagehand_layer_runtime"):
        runtime: LayerRuntime = module._stagehand_layer_runtime  # type: ignore[attr-defined]
        _mark_is_ramtorch(runtime)
        return

    # Fallback: match real RamTorch's recursive walk.
    _reattach_recursive(module)


def _reattach_recursive(module: nn.Module) -> None:
    """Walk all modules; if a module has is_ramtorch, mark its params and buffers."""
    if getattr(module, "is_ramtorch", False):
        for _name, param in module.named_parameters(recurse=False):
            if isinstance(param, torch.Tensor):
                param.is_ramtorch = True  # type: ignore[attr-defined]
        for _name, buffer in module.named_buffers(recurse=False):
            if isinstance(buffer, torch.Tensor):
                buffer.is_ramtorch = True  # type: ignore[attr-defined]
    for child in module.children():
        _reattach_recursive(child)


def _mark_is_ramtorch(runtime: LayerRuntime) -> None:
    """Set ``is_ramtorch = True`` on all managed modules, their params, and buffers."""
    for _name, mod in runtime._layer_map.items():
        mod.is_ramtorch = True  # type: ignore[attr-defined]
        for p in mod.parameters(recurse=False):
            p.is_ramtorch = True  # type: ignore[attr-defined]
        for b in mod.buffers(recurse=False):
            b.is_ramtorch = True  # type: ignore[attr-defined]


# ── Hook helpers (match ramtorch.helpers API) ────────────────────────────


def add_custom_hooks(tensor: torch.Tensor, hook_name: str = "_custom_hooks") -> torch.Tensor:
    """Add a custom hook dictionary to a tensor.

    Drop-in replacement for ``ramtorch.helpers.add_custom_hooks``.
    """
    if not hasattr(tensor, hook_name):
        setattr(tensor, hook_name, OrderedDict())
        setattr(tensor, f"{hook_name}_counter", 0)
    return tensor


def register_ramtorch_hook(tensor: torch.Tensor, hook: Callable, hook_name: str) -> int:
    """Register a hook on a tensor's custom hook dict.

    Drop-in replacement for ``ramtorch.helpers.register_ramtorch_hook``.
    """
    if not hasattr(tensor, hook_name):
        add_custom_hooks(tensor, hook_name)

    hooks = getattr(tensor, hook_name)
    counter_name = f"{hook_name}_counter"
    counter = getattr(tensor, counter_name)

    hook_id = counter
    hooks[hook_id] = hook
    setattr(tensor, counter_name, counter + 1)
    return hook_id


def register_ramtorch_grad_hook(
    module: nn.Module,
    hook_fn: Callable,
    param_names: Sequence[str] | None = None,
) -> list:
    """Register backward hooks on module parameters.

    Drop-in replacement for ``ramtorch.helpers.register_ramtorch_grad_hook``.
    For stagehand-managed params (``is_ramtorch=True``), hooks are stored in
    ``_ramtorch_backward_hooks`` on the tensor. For regular params, uses
    PyTorch's native ``register_hook``.
    """
    handles = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param_names is not None and name not in param_names:
            continue
        if getattr(param, "is_ramtorch", False):
            handle = register_ramtorch_hook(param, hook_fn, "_ramtorch_backward_hooks")
        else:
            handle = param.register_hook(hook_fn)
        handles.append(handle)
    return handles


def register_ramtorch_post_accumulate_grad_hook(
    module: nn.Module,
    hook_fn: Callable,
    param_names: Sequence[str] | None = None,
) -> list:
    """Register post-accumulate gradient hooks on module parameters.

    Drop-in replacement for
    ``ramtorch.helpers.register_ramtorch_post_accumulate_grad_hook``.
    For stagehand-managed params, hooks are stored in
    ``_ramtorch_post_accumulate_grad_hooks``. For regular params, uses
    PyTorch's native ``register_post_accumulate_grad_hook``.
    """
    handles = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param_names is not None and name not in param_names:
            continue
        if getattr(param, "is_ramtorch", False):
            handle = register_ramtorch_hook(
                param, hook_fn, "_ramtorch_post_accumulate_grad_hooks",
            )
        else:
            handle = param.register_post_accumulate_grad_hook(hook_fn)
        handles.append(handle)
    return handles


# ── Linear stub ──────────────────────────────────────────────────────────


class Linear(nn.Linear):
    """Stub for ``isinstance`` checks against ``ramtorch.Linear``.

    This class is NOT used for computation — stagehand manages the
    original ``nn.Linear`` modules via hooks. It exists solely so that
    ``isinstance(module, ramtorch.Linear)`` checks work in downstream code
    that imports Linear from this shim.
    """

    is_ramtorch: bool = True
