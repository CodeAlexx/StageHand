"""Basic Stagehand usage example.

This example is intentionally minimal and shows runtime wiring only.
"""

from __future__ import annotations

import torch
from torch import nn

from stagehand import StagehandConfig, StagehandRuntime


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])


model = TinyModel()
config = StagehandConfig(pinned_pool_mb=256, pinned_slab_mb=64)

runtime = StagehandRuntime(
    model=model,
    config=config,
    block_pattern=r"block\.\d+",
    group="example",
    dtype=torch.bfloat16,
    inference_mode=True,
)

print(f"registered_blocks={len(runtime._registry)}")
