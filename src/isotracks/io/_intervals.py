"""Restricting a per-position track to the exons a figure actually draws."""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, List, Optional, Sequence, Tuple

Interval = Tuple[int, int]


def merge_intervals(spans: Iterable[Interval]) -> List[Interval]:
    """Sort and union closed ``(start, end)`` spans."""
    ordered = sorted((int(a), int(b)) for a, b in spans if b >= a)
    merged: List[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class IntervalMask:
    """Membership test over a merged set of closed intervals."""

    def __init__(self, spans: Optional[Sequence[Interval]]):
        self.spans = merge_intervals(spans) if spans else []
        self._starts = [start for start, _ in self.spans]

    def __bool__(self) -> bool:
        """False when there is nothing to restrict to, so callers can skip."""
        return bool(self.spans)

    def contains(self, position: int) -> bool:
        if not self.spans:
            return True  # no mask means no restriction
        index = bisect_right(self._starts, position) - 1
        return index >= 0 and position <= self.spans[index][1]
