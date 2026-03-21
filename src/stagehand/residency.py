"""Residency map — tracks per-block state on the host/GPU boundary.

Each block registered in the :class:`BlockRegistry` gets a
:class:`ResidencyEntry` that records its current :class:`BlockState`,
GPU tensor reference, host slab reference, refcount, and scheduling
metadata.

State transitions are enforced by a finite-state machine.  Invalid
transitions raise :class:`InvalidStateTransitionError`.

First target model: WAN 2.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch

from stagehand.errors import InvalidStateTransitionError

if TYPE_CHECKING:
    from stagehand.pool import PinnedSlab
    from stagehand.registry import BlockRegistry

__all__ = ["BlockState", "ResidencyEntry", "ResidencyMap"]


# ── state machine ────────────────────────────────────────────────────────


class BlockState(str, Enum):
    """Lifecycle states for a swappable block."""

    UNLOADED = "unloaded"
    HOST_STAGED = "host_staged"
    PREFETCHING = "prefetching"
    GPU_READY = "gpu_ready"
    EVICTING = "evicting"
    GPU_FREEING = "gpu_freeing"


# Legal state transitions (source -> set of allowed destinations).
VALID_TRANSITIONS: dict[BlockState, frozenset[BlockState]] = {
    BlockState.UNLOADED: frozenset({BlockState.HOST_STAGED}),
    BlockState.HOST_STAGED: frozenset({BlockState.PREFETCHING}),
    BlockState.PREFETCHING: frozenset({BlockState.GPU_READY}),
    BlockState.GPU_READY: frozenset({BlockState.EVICTING, BlockState.GPU_FREEING}),
    BlockState.EVICTING: frozenset({BlockState.HOST_STAGED, BlockState.UNLOADED}),
    BlockState.GPU_FREEING: frozenset({BlockState.UNLOADED}),
}


# ── entry ────────────────────────────────────────────────────────────────


@dataclass
class ResidencyEntry:
    """Mutable per-block residency metadata."""

    state: BlockState = BlockState.UNLOADED
    gpu_tensor: torch.Tensor | None = None
    host_slab: PinnedSlab | None = None
    refcount: int = 0
    last_used_step: int = -1
    next_use_step: int | None = None
    transfer_event: torch.cuda.Event | None = None
    param_layout: list[tuple[str, tuple[int, ...], torch.dtype, int, int]] | None = None
    """Flattened parameter layout: list of (param_name, shape, dtype, offset_bytes, num_elements).

    Populated when a block is staged to host.  Used to reconstruct individual
    parameter tensors as views into the contiguous slab/GPU buffer."""


# ── map ──────────────────────────────────────────────────────────────────


class ResidencyMap:
    """Maps every registered block to its residency state.

    Created from a :class:`BlockRegistry`; one :class:`ResidencyEntry` per
    block.  All state mutations go through :meth:`transition`, which
    enforces the finite-state machine defined by :data:`VALID_TRANSITIONS`.
    """

    def __init__(self, registry: BlockRegistry) -> None:
        self._entries: dict[str, ResidencyEntry] = {}
        for entry in registry.blocks_in_order():
            self._entries[entry.block_id] = ResidencyEntry()

    # ── state queries ────────────────────────────────────────────────

    def get_state(self, block_id: str) -> BlockState:
        """Current state of *block_id*."""
        return self._entries[block_id].state

    def get_entry(self, block_id: str) -> ResidencyEntry:
        """Full residency entry for *block_id*."""
        return self._entries[block_id]

    # ── state transitions ────────────────────────────────────────────

    def transition(self, block_id: str, new_state: BlockState) -> None:
        """Move *block_id* to *new_state*.

        Raises
        ------
        InvalidStateTransitionError
            If the transition is not in :data:`VALID_TRANSITIONS`.
        KeyError
            If *block_id* is unknown.
        """
        entry = self._entries[block_id]
        allowed = VALID_TRANSITIONS.get(entry.state, frozenset())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition {block_id!r} from {entry.state.value!r} "
                f"to {new_state.value!r}"
            )
        entry.state = new_state

    # ── refcount ─────────────────────────────────────────────────────

    def increment_ref(self, block_id: str) -> None:
        """Increment the refcount (block is in use by compute)."""
        self._entries[block_id].refcount += 1

    def decrement_ref(self, block_id: str) -> None:
        """Decrement the refcount."""
        entry = self._entries[block_id]
        if entry.refcount <= 0:
            raise ValueError(f"Refcount for {block_id!r} is already {entry.refcount}")
        entry.refcount -= 1

    # ── eviction helpers ─────────────────────────────────────────────

    def can_evict(self, block_id: str) -> bool:
        """True if *block_id* is GPU_READY with refcount == 0."""
        entry = self._entries[block_id]
        return entry.state == BlockState.GPU_READY and entry.refcount == 0

    def gpu_resident_blocks(self) -> list[str]:
        """All block_ids currently in GPU_READY state."""
        return [
            bid
            for bid, entry in self._entries.items()
            if entry.state == BlockState.GPU_READY
        ]

    def eviction_candidates(
        self,
        current_step: int,
        cooldown_steps: int,
    ) -> list[tuple[str, ResidencyEntry]]:
        """Blocks eligible for eviction.

        Returns blocks that are GPU_READY, have refcount == 0, and whose
        ``last_used_step`` is older than *current_step* - *cooldown_steps*.
        Results are returned in no particular order; the caller (scheduler)
        is responsible for scoring and sorting.
        """
        threshold = current_step - cooldown_steps
        candidates: list[tuple[str, ResidencyEntry]] = []
        for bid, entry in self._entries.items():
            if (
                entry.state == BlockState.GPU_READY
                and entry.refcount == 0
                and entry.last_used_step <= threshold
            ):
                candidates.append((bid, entry))
        return candidates

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._entries

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.state.value] = counts.get(entry.state.value, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"ResidencyMap({summary})"
