from __future__ import annotations

import stagehand
from stagehand import StagehandConfig


def test_package_imports() -> None:
    assert hasattr(stagehand, "__version__")
    cfg = StagehandConfig()
    assert cfg.pinned_pool_mb > 0
