"""BAM readers: per-base coverage, splice junctions and read-to-feature conversion."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pysam

from .features import Feature, FeatureArray

DEPTH_COLUMNS = ["pos", "depth"]
JUNCTION_COLUMNS = ["start", "end", "count", "left_depth", "right_depth", "ratio"]


def empty_depth() -> pd.DataFrame:
    return pd.DataFrame([], columns=DEPTH_COLUMNS)


def empty_junctions() -> pd.DataFrame:
    return pd.DataFrame([], columns=JUNCTION_COLUMNS)


def read_blocks(read: pysam.AlignedSegment) -> List[Tuple[int, int]]:
    """Aligned blocks of a read, 1-based inclusive, split only at ``N`` (splice)."""
    blocks: List[Tuple[int, int]] = []
    start = read.reference_start + 1
    end = start - 1
    for op, length in read.cigartuples or ():
        if op in (1, 4, 5):  # insertion, soft clip, hard clip
            continue
        if op in (0, 2, 7, 8):  # match, deletion, =, X
            end += length
        elif op == 3:  # N: splice junction
            blocks.append((start, end))
            start = end + length + 1
            end = start - 1
    blocks.append((start, end))
    return blocks


def read_introns(read: pysam.AlignedSegment, as_junction: bool = False) -> List[Tuple[int, int]]:
    """Intron spans of a read, 1-based."""
    if read.is_unmapped:
        return []
    blocks = read_blocks(read)
    introns = []
    for (_, left_end), (right_start, _) in zip(blocks[:-1], blocks[1:]):
        if as_junction:
            introns.append((left_end, right_start))
        else:
            introns.append((left_end + 1, right_start - 1))
    return introns


def read_exons_and_introns(read: pysam.AlignedSegment) -> List[Tuple[Tuple[int, int], str]]:
    """Alternating exon/intron spans of a read, in genomic order."""
    blocks = read_blocks(read)
    introns = read_introns(read)
    parts: List[Tuple[Tuple[int, int], str]] = []
    for block, intron in zip(blocks[:-1], introns):
        parts.append((block, "exon"))
        parts.append((intron, "intron"))
    parts.append((blocks[-1], "exon"))
    return parts


def read_to_feature(read: pysam.AlignedSegment, type_name: str = "mRNA") -> Feature:
    """Convert an aligned read into a :class:`Feature` with exon/intron children."""
    start = read.reference_start + 1
    end = read.reference_end
    chr_name = read.reference_name
    strand = "-" if read.is_reverse else "+"
    feature_id = f"{read.query_name},{chr_name},{start},{end},{strand}"
    children = [
        Feature(block_start, block_end, chr_name, strand, "", 1, block_type)
        for (block_start, block_end), block_type in read_exons_and_introns(read)
    ]
    return Feature(start, end, chr_name, strand, feature_id, 1, type_name,
                   parm={"read": read}, children=children)


def reads_to_features(reads: Iterable[pysam.AlignedSegment], type_name: str = "mRNA") -> FeatureArray:
    features = [read_to_feature(r, type_name) for r in reads if not r.is_unmapped]
    return FeatureArray(features)


class BamReader:
    """Base for readers that stream a region out of one or more BAM files."""

    def __init__(
        self,
        file_names: Optional[Sequence[str]] = None,
        file_name: str = "",
        min_mapping_quality: int = 1,
        strand_filter: Optional[str] = None,
        read_filter: Optional[Callable[[pysam.AlignedSegment], bool]] = None,
    ):
        if file_names is None:
            file_names = [file_name] if file_name else []
        self.file_names = list(file_names)
        self.min_mapping_quality = min_mapping_quality
        self.strand_filter = strand_filter
        self.read_filter = read_filter
        self.handles: List[pysam.AlignmentFile] = []
        self.region_data = None

    def open(self) -> None:
        self.close()
        self.handles = [pysam.AlignmentFile(name, "rb") for name in self.file_names]

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.close()
            except (ValueError, OSError):
                pass
        self.handles = []

    def keep(self, read: pysam.AlignedSegment) -> bool:
        """Whether a read should contribute to the track."""
        if read.is_secondary or read.is_supplementary or read.is_unmapped:
            return False
        if read.mapping_quality < self.min_mapping_quality:
            return False
        if self.strand_filter is not None:
            if ("-" if read.is_reverse else "+") != self.strand_filter:
                return False
        if self.read_filter is not None and not self.read_filter(read):
            return False
        return True

    def iter_region_reads(self) -> Iterable[pysam.AlignedSegment]:
        for handle in self.handles:
            for read in handle.fetch(self.chr_name, self.region_start - 1, self.region_end):
                if self.keep(read):
                    yield read

    def load_region_data(self, region: Sequence, select_features=None):
        chr_name, region_start, region_end, region_strand = region
        self.chr_name = chr_name
        self.region_start = int(region_start)
        self.region_end = int(region_end)
        self.region_strand = region_strand
        self.open()
        try:
            self.region_data = self._read_region()
        finally:
            self.close()
        return self.region_data

    def _read_region(self):  # pragma: no cover - overridden
        raise NotImplementedError


class CoverageReader(BamReader):
    """Per-base depth and splice-junction counts across a set of BAM files."""

    def __init__(self, *args, min_junction_overlap: int = 0, **kwargs):
        self.min_junction_overlap = min_junction_overlap
        super().__init__(*args, **kwargs)
        self.region_data = [empty_depth(), empty_junctions()]

    @staticmethod
    def _add_read_depth(read, depth: np.ndarray, left: int, right: int) -> None:
        for block_start, block_end in read_blocks(read):
            lo = max(block_start, left)
            hi = min(block_end, right)
            if lo <= hi:
                depth[lo - left:hi - left + 1] += 1

    @staticmethod
    def _junction_frame(counts: Counter) -> pd.DataFrame:
        if not counts:
            return empty_junctions()
        rows = [[start, end, count] for (start, end), count in sorted(counts.items())]
        return pd.DataFrame(rows, columns=["start", "end", "count"])

    @staticmethod
    def _attach_depth(junctions: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
        if junctions.empty:
            return empty_junctions()
        lookup = depth.set_index("pos")["depth"] if not depth.empty else pd.Series(dtype=int)
        junctions = junctions.copy()
        junctions["left_depth"] = junctions["start"].map(lookup).fillna(0)
        junctions["right_depth"] = junctions["end"].map(lookup).fillna(0)
        total = junctions["left_depth"] + junctions["right_depth"]
        junctions["ratio"] = np.where(total > 0, junctions["count"] * 2 / total, 0.0)
        return junctions

    def _read_region(self):
        counts: Counter = Counter()
        left, right = self.region_start, self.region_end

        for read in self.iter_region_reads():
            for intron_start, intron_end in read_introns(read, as_junction=True):
                counts[(intron_start, intron_end)] += 1
                left = min(left, intron_start)
                right = max(right, intron_end)

        positions = np.arange(left, right + 1)
        depth = np.zeros(len(positions), dtype=int)
        for read in self.iter_region_reads():
            self._add_read_depth(read, depth, left, right)

        depth_df = pd.DataFrame({"pos": positions, "depth": depth})
        junctions = self._attach_depth(self._junction_frame(counts), depth_df)
        in_window = (depth_df["pos"] >= self.region_start) & (depth_df["pos"] <= self.region_end)
        return [depth_df[in_window], junctions]
