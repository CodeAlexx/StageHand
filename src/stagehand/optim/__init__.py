"""Optimizers designed for use with stagehand CPU offloading."""
from __future__ import annotations

from stagehand.optim.offloaded_adamw import OffloadedAdamW

__all__ = ["OffloadedAdamW"]
