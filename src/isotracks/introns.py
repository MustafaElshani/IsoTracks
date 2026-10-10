"""Shortened introns: a piecewise-linear map from genomic to drawn position."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .io._intervals import merge_intervals

#: Allowed values of ``introns``.
INTRON_GENOMIC = "genomic"
INTRON_SCALED = "scaled"
INTRON_FIXED = "fixed"
INTRON_MODES = (INTRON_GENOMIC, INTRON_SCALED, INTRON_FIXED)

#: Share of the drawn width a single intron gets in ``"fixed"`` mode.
FIXED_INTRON_SHARE = 0.04
#: Most of the width all introns may take together in ``"fixed"`` mode.
MAX_FIXED_TOTAL = 0.5
#: Least width any intron is drawn at in ``"scaled"`` mode, as a share of the whole.
MIN_INTRON_SHARE = 0.012
#: Slope 1 beyond the outermost knots.
_FAR = 1e12


@dataclass
class IntronMap:
    """Genomic position to drawn position, and back."""

    genomic: np.ndarray
    drawn: np.ndarray
    mode: str
    breaks: List[Tuple[int, int]] = field(default_factory=list)

    def forward(self, values):
        return np.interp(np.asarray(values, dtype=float), self.genomic, self.drawn)

    def inverse(self, values):
        return np.interp(np.asarray(values, dtype=float), self.drawn, self.genomic)

    @property
    def functions(self):
        """The pair ``ax.set_xscale("function", functions=...)`` takes."""
        return self.forward, self.inverse


def build_intron_map(exons: Sequence[Tuple[int, int]], window: Sequence,
                     mode: str, share: float = 0.3) -> Optional[IntronMap]:
    """A map shortening the introns between *exons* within *window*."""
    if mode not in INTRON_MODES:
        raise ValueError(f"unknown introns mode {mode!r}; expected one of "
                         + ", ".join(repr(m) for m in INTRON_MODES))
    if mode == INTRON_GENOMIC:
        return None
    if not 0.05 <= share <= 0.9:
        raise ValueError(f"intron share must be between 0.05 and 0.9, got {share}")

    low, high = sorted((float(window[1]), float(window[2])))
    blocks = [(max(s - 0.5, low), min(e + 0.5, high))
              for s, e in merge_intervals(exons) if e + 0.5 > low and s - 0.5 < high]
    introns = [(a_end, b_start) for (_, a_end), (b_start, _) in zip(blocks, blocks[1:])
               if b_start > a_end]
    if not introns:
        return None

    intronic = sum(b - a for a, b in introns)
    kept = (high - low) - intronic  # exons and flanks, at true scale
    if kept <= 0:
        return None

    if mode == INTRON_SCALED:
        target = share / (1.0 - share) * kept
        if intronic <= target:
            return None
        factor = target / intronic
        floor = MIN_INTRON_SHARE * (kept + target)
        widths = [min(b - a, max((b - a) * factor, floor)) for a, b in introns]
    else:
        each = min(FIXED_INTRON_SHARE, MAX_FIXED_TOTAL / len(introns))
        width = each * kept / (1.0 - each * len(introns))
        widths = [min(b - a, width) for a, b in introns]

    breaks = [(int(round(a + 0.5)), int(round(b - 0.5)))
              for (a, b), w in zip(introns, widths) if w < (b - a) - 1e-6]
    if not breaks:
        return None

    genomic, drawn = [low], [low]

    def knot(g: float, d: float) -> None:
        if g > genomic[-1]:
            genomic.append(g)
            drawn.append(d)

    at_g = at_d = low
    for (a, b), width in zip(introns, widths):
        at_d += a - at_g
        at_g = a
        knot(at_g, at_d)
        at_d += width
        at_g = b
        knot(at_g, at_d)
    knot(high, at_d + (high - at_g))

    # Slope 1 beyond the window on both sides.
    left = drawn[0] - genomic[0]
    right = drawn[-1] - genomic[-1]
    genomic = [-_FAR] + genomic + [_FAR]
    drawn = [-_FAR + left] + drawn + [_FAR + right]
    return IntronMap(np.asarray(genomic), np.asarray(drawn), mode, breaks)
