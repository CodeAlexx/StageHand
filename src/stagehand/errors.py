"""Custom exceptions for the stagehand block-swapping runtime."""
from __future__ import annotations

__all__ = [
    "StagehandError",
    "StagehandOOMError",
    "DtypeMismatchError",
    "InvalidStateTransitionError",
    "TransferError",
]


class StagehandError(Exception):
    """Base exception for all stagehand errors."""


class StagehandOOMError(StagehandError):
    """Block too large for the pinned pool."""


class DtypeMismatchError(StagehandError):
    """Strict bf16 enforcement violation."""


class InvalidStateTransitionError(StagehandError):
    """Illegal residency state machine transition."""


class TransferError(StagehandError):
    """Async H2D/D2H copy failure."""
