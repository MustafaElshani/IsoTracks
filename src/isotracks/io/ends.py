"""Where reads start and stop: the evidence behind a TSS or poly(A) call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Set

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .longreads import RegionReadPool, read_key

#: Gaussian bandwidth in base pairs.
DEFAULT_BANDWIDTH = 25.0

#: Densities are reported per this many base pairs.
PER_BASES = 100.0


@dataclass
class EndDensity:
    """Smoothed read-end density across a region."""

    positions: np.ndarray
    five_prime: np.ndarray
    three_prime: np.ndarray
    five_prime_reads: int
    three_prime_reads: int

    @property
    def peak(self) -> float:
        """Highest density either curve reaches, in reads per 100 bp."""
        return float(max(
            self.five_prime.max(initial=0.0),
            self.three_prime.max(initial=0.0),
        ))

    def is_empty(self) -> bool:
        return self.five_prime_reads == 0 and self.three_prime_reads == 0


class ReadEndReader:
    """Counts read 5' and 3' ends over a region and smooths them.

    Parameters
    ----------
    pools : sequence of RegionReadPool
        Region reads to count, one pool per condition.
    allowed_read_names : set of str, optional
        Restrict to these reads, matched on read id.
    bandwidth : float
        Gaussian bandwidth in base pairs.
    """

    def __init__(
        self,
        pools: Sequence[RegionReadPool],
        allowed_read_names: Optional[Sequence[str]] = None,
        bandwidth: float = DEFAULT_BANDWIDTH,
    ):
        self.pools = list(pools)
        self.allowed: Optional[Set[str]] = (
            {read_key(name) for name in allowed_read_names} if allowed_read_names else None
        )
        self.bandwidth = float(bandwidth)
        self.region_data: Optional[EndDensity] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> EndDensity:
        self.region_data = self.read_region(region)
        return self.region_data

    def read_region(self, region: Sequence) -> EndDensity:
        chr_name, start, end, _strand = region
        start, end = int(start), int(end)
        fives: List[int] = []
        threes: List[int] = []
        for pool in self.pools:
            for read in pool.reads(chr_name, start, end):
                if self.allowed is not None and read_key(read.query_name) not in self.allowed:
                    continue
                low = read.reference_start + 1
                high = read.reference_end
                if low is None or high is None:
                    continue
                five_pos, three_pos = (high, low) if read.is_reverse else (low, high)
                fives.append(five_pos)
                threes.append(three_pos)
        return smoothed_ends(start, end, fives, threes, self.bandwidth)


class EndPositionReader:
    """Read-end density from ends already known, by chromosome."""

    def __init__(self, five_prime: Mapping[str, Sequence[int]],
                 three_prime: Mapping[str, Sequence[int]],
                 bandwidth: float = DEFAULT_BANDWIDTH):
        self.five_prime = {c: np.asarray(v, dtype=int) for c, v in five_prime.items()}
        self.three_prime = {c: np.asarray(v, dtype=int) for c, v in three_prime.items()}
        self.bandwidth = float(bandwidth)
        self.region_data: Optional[EndDensity] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> EndDensity:
        self.region_data = self.read_region(region)
        return self.region_data

    def read_region(self, region: Sequence) -> EndDensity:
        chr_name, start, end = region[0], int(region[1]), int(region[2])
        empty = np.zeros(0, dtype=int)
        return smoothed_ends(start, end, self.five_prime.get(chr_name, empty),
                             self.three_prime.get(chr_name, empty), self.bandwidth)


def smoothed_ends(start: int, end: int, fives: Sequence[int], threes: Sequence[int],
                  bandwidth: float = DEFAULT_BANDWIDTH) -> EndDensity:
    """Count the ends inside ``start..end`` per base and smooth each count."""
    width = max(end - start + 1, 1)
    curves, counts = [], []
    for positions in (fives, threes):
        index = np.asarray(positions, dtype=int) - start
        index = index[(index >= 0) & (index < width)]
        curve = np.bincount(index, minlength=width).astype(float)
        if bandwidth > 0:
            curve = gaussian_filter1d(curve, bandwidth, mode="constant")
        curves.append(curve * PER_BASES)
        counts.append(int(len(index)))
    return EndDensity(
        positions=np.arange(start, start + width),
        five_prime=curves[0],
        three_prime=curves[1],
        five_prime_reads=counts[0],
        three_prime_reads=counts[1],
    )
