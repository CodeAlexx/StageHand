from __future__ import annotations

import pytest

from stagehand.budget import BudgetManager


class TestBudgetManager:
    def test_init(self) -> None:
        b = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        assert b.high_watermark_mb == 22000
        assert b.low_watermark_mb == 18000

    def test_invalid_watermarks(self) -> None:
        with pytest.raises(ValueError):
            BudgetManager(high_watermark_mb=1000, low_watermark_mb=1000)

    def test_headroom_numeric(self) -> None:
        b = BudgetManager(high_watermark_mb=22000, low_watermark_mb=18000)
        value = b.headroom_mb()
        assert isinstance(value, float)
