"""Differential modification from ``modkit dmr pair``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pysam

from ._intervals import IntervalMask

#: Columns of a ``modkit dmr pair`` site table, by position.
CHROM, START, END, STRAND = 0, 1, 2, 5
A_TOTAL, B_TOTAL = 7, 9
A_PCT, B_PCT = 12, 13
P_VALUE, EFFECT_SIZE = 14, 15
BALANCED_P, BALANCED_EFFECT = 16, 17
#: A row must have at least this many fields to be a dmr-pair row.
MIN_FIELDS = 18

SITE_COLUMNS = [
    "chrom", "position", "strand", "a_fraction", "b_fraction",
    "effect_size", "p_value", "a_total", "b_total",
]


class ModDiffInputError(ValueError):
    """A dmr-pair table is missing, or has no tabix index."""


def check_diff_tables(tables: Dict[str, str]) -> None:
    """Fail early and legibly on a missing file or a missing index."""
    missing, unindexed = [], []
    for label, path in tables.items():
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
        elif not os.path.exists(path + ".tbi") and not os.path.exists(path + ".csi"):
            unindexed.append(f"  {label}: {path}")

    problems = []
    if missing:
        problems.append("dmr pair table(s) not found:\n" + "\n".join(missing))
    if unindexed:
        problems.append(
            "dmr pair table(s) have no tabix index:\n" + "\n".join(unindexed)
            + "\n\nSort, compress and index with:\n"
            "   grep -v '^#' sites.bed | sort -k1,1 -k2,2n | bgzip > sites.bed.gz\n"
            "   tabix -p bed sites.bed.gz\n"
            "The sort is required: modkit writes a chromosome's rows in several\n"
            "blocks, and tabix rejects that with 'Chromosome blocks not continuous'."
        )
    if problems:
        raise ModDiffInputError("\n\n".join(problems))


@dataclass
class DiffSites:
    """Differential modification calls in a region."""

    frame: pd.DataFrame
    tested: int = 0  # sites that met the coverage floor, before filtering

    def is_empty(self) -> bool:
        return self.frame.empty

    @property
    def lost(self) -> pd.DataFrame:
        """Sites less modified in the second sample set."""
        return self.frame[self.frame["effect_size"] > 0]

    @property
    def gained(self) -> pd.DataFrame:
        """Sites more modified in the second sample set."""
        return self.frame[self.frame["effect_size"] < 0]


class ModDiffReader:
    """A ``modkit dmr pair`` site table, filtered to a region on read.

    Parameters
    ----------
    file_name : str
        Tabix-indexed dmr-pair output.
    min_coverage : int
        Both sides must have been assessed in at least this many reads.
    max_p_value : float
        Sites less significant than this are dropped.
    min_effect : float
        Sites whose modification changed by less than this fraction are
        dropped.
    balanced : bool
        Use modkit's replicate-balanced effect size and p-value.
    strand_filter : str, optional
        Keep only calls on this strand.
    regions : sequence of (start, end), optional
        Keep only positions inside these closed spans.
    """

    def __init__(
        self,
        file_name: str,
        min_coverage: int = 10,
        max_p_value: float = 0.05,
        min_effect: float = 0.2,
        balanced: bool = False,
        strand_filter: Optional[str] = None,
        regions: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.file_name = file_name
        self.min_coverage = int(min_coverage)
        self.max_p_value = float(max_p_value)
        self.min_effect = float(min_effect)
        self.balanced = balanced
        self.strand_filter = strand_filter
        self.mask = IntervalMask(regions)
        self.region_data: Optional[DiffSites] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> DiffSites:
        self.region_data = self.read_region(region)
        return self.region_data

    def _rows(self, chrom: str, start: int, end: int) -> Iterable[List[str]]:
        with pysam.TabixFile(self.file_name) as handle:
            if chrom not in handle.contigs:
                return
            for line in handle.fetch(chrom, max(start - 1, 0), end):
                yield line.split("\t")

    def read_region(self, region: Sequence) -> DiffSites:
        chrom, start, end, _strand = region
        start, end = int(start), int(end)
        effect_column = BALANCED_EFFECT if self.balanced else EFFECT_SIZE
        p_column = BALANCED_P if self.balanced else P_VALUE

        rows: List[dict] = []
        tested = 0
        for fields in self._rows(chrom, start, end):
            if len(fields) < MIN_FIELDS:
                continue
            strand = fields[STRAND]
            if self.strand_filter is not None and strand != self.strand_filter:
                continue
            try:
                position = int(fields[START]) + 1  # the table is 0-based
                a_total, b_total = int(fields[A_TOTAL]), int(fields[B_TOTAL])
                a_fraction, b_fraction = float(fields[A_PCT]), float(fields[B_PCT])
                effect = float(fields[effect_column])
                p_value = float(fields[p_column])
            except ValueError:
                continue

            if not self.mask.contains(position):
                continue
            if min(a_total, b_total) < self.min_coverage:
                continue
            tested += 1
            if p_value > self.max_p_value or abs(effect) < self.min_effect:
                continue
            rows.append({
                "chrom": chrom, "position": position, "strand": strand,
                "a_fraction": a_fraction, "b_fraction": b_fraction,
                "effect_size": effect, "p_value": p_value,
                "a_total": a_total, "b_total": b_total,
            })

        frame = pd.DataFrame(rows, columns=SITE_COLUMNS)
        if not frame.empty:
            frame = frame.sort_values("position").reset_index(drop=True)
        return DiffSites(frame=frame, tested=tested)
