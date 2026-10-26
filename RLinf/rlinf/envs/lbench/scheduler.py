"""Deterministic task-balanced scheduling helpers for LBench."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


class BalancedTaskScheduler:
    """Round-robin tasks globally across ranks, env slots and episodes."""

    def __init__(self, task_indices: Sequence[int]):
        self.task_indices = tuple(int(index) for index in task_indices)
        if not self.task_indices or len(set(self.task_indices)) != len(
            self.task_indices
        ):
            raise ValueError("balanced task indices must be non-empty and unique")

    def task_for_ordinal(self, ordinal: int) -> int:
        if ordinal < 0:
            raise ValueError("global episode ordinal must be non-negative")
        return self.task_indices[ordinal % len(self.task_indices)]

    def counts(self, start: int, count: int) -> dict[int, int]:
        if start < 0 or count < 0:
            raise ValueError("start and count must be non-negative")
        return dict(
            Counter(self.task_for_ordinal(start + offset) for offset in range(count))
        )


__all__ = ["BalancedTaskScheduler"]
