"""RNA modification pileups from ``modkit pileup`` bedMethyl."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pysam

from ._intervals import IntervalMask

#: Modification codes in column 4 of a bedMethyl file, mapped to their labels.
MOD_NAMES: Dict[str, str] = {
    "a": "m6A",  # N6-methyladenosine
    "m": "5mC",  # 5-methylcytosine
    "h": "5hmC",  # 5-hydroxymethylcytosine
    "17596": "inosine",
    "17802": "pseU",  # pseudouridine
    "69426": "Am",  # 2'-O-methyladenosine
    "19228": "Cm",  # 2'-O-methylcytidine
    "19229": "Gm",  # 2'-O-methylguanosine
    "19227": "Um",  # 2'-O-methyluridine
}

#: The bedMethyl columns this reader uses, by position.
CHROM, START, CODE, STRAND = 0, 1, 3, 5
VALID_COVERAGE, PERCENT_MODIFIED, N_MODIFIED = 9, 10, 11

SITE_COLUMNS = ["chrom", "position", "code", "strand", "coverage", "modified", "fraction"]


def mod_label(code: str) -> str:
    """How to write a modification code in a figure."""
    return MOD_NAMES.get(str(code), str(code))


class ModkitInputError(ValueError):
    """A bedMethyl file is missing, or has no tabix index."""


def check_pileups(pileups: Dict[str, Sequence[str]]) -> None:
    """Fail early and legibly on a missing file or a missing index."""
    missing, unindexed = [], []
    for condition, paths in pileups.items():
        for path in paths:
            if not os.path.exists(path):
                missing.append(f"  {condition}: {path}")
            elif not os.path.exists(path + ".tbi") and not os.path.exists(path + ".csi"):
                unindexed.append(f"  {condition}: {path}")

    problems = []
    if missing:
        problems.append("modkit pileup file(s) not found:\n" + "\n".join(missing))
    if unindexed:
        problems.append(
            "modkit pileup file(s) have no tabix index:\n" + "\n".join(unindexed)
            + "\n\nIndex them with:  bgzip -k pileup.bed && tabix -p bed pileup.bed.gz"
        )
    if problems:
        raise ModkitInputError("\n\n".join(problems))


@dataclass
class ModSites:
    """Modification calls in a region, one row per position per code."""

    frame: pd.DataFrame
    assessed: int = 0  # sites that met the coverage floor, before filtering

    def is_empty(self) -> bool:
        return self.frame.empty

    def codes(self) -> List[str]:
        if self.frame.empty:
            return []
        return list(self.frame["code"].unique())

    def for_code(self, code: str) -> pd.DataFrame:
        return self.frame[self.frame["code"] == code]


class ModPileupReader:
    """Pooled modification pileup over one condition's bedMethyl files.

    Parameters
    ----------
    file_names : sequence of str
        Tabix-indexed bedMethyl files, one per replicate.
    codes : sequence of str, optional
        Modification codes to keep.
    min_coverage : int
        Sites assessed in fewer reads than this are dropped.
    min_fraction : float
        Sites modified in less than this fraction of reads are dropped.
    strand_filter : str, optional
        Keep only calls on this strand.
    regions : sequence of (start, end), optional
        Keep only positions inside these closed spans.
    """

    def __init__(
        self,
        file_names: Sequence[str],
        codes: Optional[Sequence[str]] = None,
        min_coverage: int = 10,
        min_fraction: float = 0.1,
        strand_filter: Optional[str] = None,
        regions: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.file_names = list(file_names)
        self.codes = [str(c) for c in codes] if codes else None
        self.min_coverage = int(min_coverage)
        self.min_fraction = float(min_fraction)
        self.strand_filter = strand_filter
        self.mask = IntervalMask(regions)
        self.region_data: Optional[ModSites] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> ModSites:
        self.region_data = self.read_region(region)
        return self.region_data

    def _rows(self, path: str, chrom: str, start: int, end: int) -> Iterable[List[str]]:
        with pysam.TabixFile(path) as handle:
            if chrom not in handle.contigs:
                return
            for line in handle.fetch(chrom, max(start - 1, 0), end):
                yield line.split("\t")

    def read_region(self, region: Sequence) -> ModSites:
        chrom, start, end, _strand = region
        start, end = int(start), int(end)

        pooled: Dict[Tuple[int, str, str], List[int]] = {}
        for path in self.file_names:
            for fields in self._rows(path, chrom, start, end):
                if len(fields) <= N_MODIFIED:
                    continue
                code = fields[CODE]
                if self.codes is not None and code not in self.codes:
                    continue
                strand = fields[STRAND]
                if self.strand_filter is not None and strand != self.strand_filter:
                    continue
                try:
                    position = int(fields[START]) + 1  # bedMethyl is 0-based
                    coverage = int(fields[VALID_COVERAGE])
                    modified = int(fields[N_MODIFIED])
                except ValueError:
                    continue
                if not self.mask.contains(position):
                    continue
                entry = pooled.setdefault((position, code, strand), [0, 0])
                entry[0] += coverage
                entry[1] += modified

        rows = []
        assessed = 0
        for (position, code, strand), (coverage, modified) in pooled.items():
            if coverage < self.min_coverage:
                continue
            assessed += 1
            fraction = modified / coverage if coverage else 0.0
            if fraction < self.min_fraction:
                continue
            rows.append({
                "chrom": chrom, "position": position, "code": code,
                "strand": strand, "coverage": coverage,
                "modified": modified, "fraction": fraction,
            })

        frame = pd.DataFrame(rows, columns=SITE_COLUMNS)
        if not frame.empty:
            frame = frame.sort_values(["code", "position"]).reset_index(drop=True)
        return ModSites(frame=frame, assessed=assessed)
