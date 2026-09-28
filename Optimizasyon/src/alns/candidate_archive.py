"""Candidate archive for ALNS candidate solutions."""

from __future__ import annotations

from typing import Dict, List, Tuple
import math
from src.alns.state import ALNSSolutionState


class CandidateArchive:
    """Bounded top-K archive for storing best candidate states sorted by proxy cost."""

    def __init__(self, capacity: int = 8):
        if capacity <= 0:
            raise ValueError("CandidateArchive capacity must be a positive integer.")
        self.capacity = capacity
        # Storage: dict signature -> (proxy_cost, state)
        self._candidates: dict[str, tuple[float, ALNSSolutionState]] = {}

    def add(self, state: ALNSSolutionState, proxy_cost: float) -> bool:
        """Add candidate state to archive if valid, non-infinite, and within top-K capacity."""
        if math.isinf(proxy_cost) or math.isnan(proxy_cost):
            return False

        sig = state.signature()
        if sig in self._candidates:
            return False

        self._candidates[sig] = (proxy_cost, state)

        # If exceeding capacity, keep only top K sorted by proxy_cost then canonical_tuple()
        if len(self._candidates) > self.capacity:
            sorted_items = sorted(
                self._candidates.items(),
                key=lambda item: (item[1][0], item[1][1].canonical_tuple())
            )
            self._candidates = dict(sorted_items[:self.capacity])
            return sig in self._candidates

        return True

    def get_candidates(self) -> list[tuple[float, ALNSSolutionState]]:
        """Return candidates sorted deterministically by proxy cost then canonical tuple."""
        return sorted(
            self._candidates.values(),
            key=lambda item: (item[0], item[1].canonical_tuple())
        )

    def clear(self) -> None:
        """Clear all stored candidates."""
        self._candidates.clear()

    def __len__(self) -> int:
        return len(self._candidates)


__all__ = ["CandidateArchive"]
