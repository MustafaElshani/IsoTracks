"""Long-read BAM reader with native Dorado poly(A) tail support."""

from __future__ import annotations

from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import pysam

from .alignments import BamReader, reads_to_features
from .features import Feature, FeatureArray

#: Tags checked, in order, for a poly(A) tail estimate.
POLYA_TAGS = ("pt", "pa")

#: Default visual length of a poly(A) tail, in genomic base pairs per nucleotide of tail.
POLYA_VISUAL_SCALE = 1.0


def polya_length_from_tags(read: pysam.AlignedSegment) -> Optional[int]:
    """Poly(A) tail length from the read's tags, or ``None`` if untagged."""
    for tag in POLYA_TAGS:
        try:
            value = read.get_tag(tag)
        except KeyError:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def read_key(read_name: str) -> str:
    """The identity of a read, independent of any pipeline prefix."""
    return read_name.rsplit("|", 1)[-1]


class RegionReadPool:
    """One fetch of a region per BAM set, shared by every track that slices it."""

    def __init__(
        self,
        file_names: Sequence[str],
        min_mapping_quality: int = 1,
        strand_filter: Optional[str] = None,
    ):
        self.file_names = list(file_names)
        self.min_mapping_quality = min_mapping_quality
        self.strand_filter = strand_filter
        self._region: Optional[tuple] = None
        self._reads: List[pysam.AlignedSegment] = []
        self._by_key: Optional[Dict[str, List[pysam.AlignedSegment]]] = None

    def _passes(self, read: pysam.AlignedSegment) -> bool:
        if read.is_secondary or read.is_supplementary or read.is_unmapped:
            return False
        if read.mapping_quality < self.min_mapping_quality:
            return False
        if self.strand_filter is not None:
            if ("-" if read.is_reverse else "+") != self.strand_filter:
                return False
        return True

    def reads(self, chr_name: str, start: int, end: int) -> List[pysam.AlignedSegment]:
        key = (chr_name, int(start), int(end))
        if key == self._region:
            return self._reads

        self.clear()
        reads: List[pysam.AlignedSegment] = []
        for name in self.file_names:
            with pysam.AlignmentFile(name, "rb") as handle:
                reads.extend(
                    read for read in handle.fetch(chr_name, int(start) - 1, int(end))
                    if self._passes(read)
                )
        self._reads = reads
        self._region = key
        return self._reads

    def reads_for(
        self, chr_name: str, start: int, end: int, keys: Optional[Sequence[str]],
    ) -> Iterator[pysam.AlignedSegment]:
        """The pooled reads whose id is in *keys*, or all of them if ``None``."""
        reads = self.reads(chr_name, start, end)
        if keys is None:
            yield from reads
            return
        if self._by_key is None:
            index: Dict[str, List[pysam.AlignedSegment]] = {}
            for read in reads:
                index.setdefault(read_key(read.query_name), []).append(read)
            self._by_key = index
        for key in keys:
            yield from self._by_key.get(key, ())

    def clear(self) -> None:
        """Drop the held region, releasing its reads."""
        self._region = None
        self._reads = []
        self._by_key = None


class LongReadBam(BamReader):
    """Nanopore alignments as poly(A)-aware features.

    Parameters
    ----------
    allowed_read_names : set of str, optional
        Only these reads are kept.
    polya_lengths : mapping, optional
        Fallback tail lengths keyed by read name, used when the BAM has no
        ``pt``/``pa`` tag.
    min_polya_length : int
        Tails shorter than this are not drawn.
    polya_scale : float
        Genomic base pairs drawn per nucleotide of tail.
    pool : RegionReadPool, optional
        Share one fetch of the region with the other readers drawing it.
    """

    def __init__(
        self,
        *args,
        allowed_read_names: Optional[Sequence[str]] = None,
        polya_lengths: Optional[Mapping[str, int]] = None,
        min_polya_length: int = 15,
        draw_polya: bool = True,
        polya_scale: float = POLYA_VISUAL_SCALE,
        pool: Optional["RegionReadPool"] = None,
        **kwargs,
    ):
        self.allowed_read_names = (
            {read_key(name) for name in allowed_read_names} if allowed_read_names else None
        )
        self.polya_lengths: Mapping[str, int] = (
            {read_key(name): value for name, value in polya_lengths.items()}
            if polya_lengths else {}
        )
        self.min_polya_length = min_polya_length
        self.draw_polya = draw_polya
        self.polya_scale = float(polya_scale)
        self.pool = pool
        super().__init__(*args, **kwargs)
        if pool is not None and not self.file_names:
            self.file_names = list(pool.file_names)

    def keep(self, read: pysam.AlignedSegment) -> bool:
        if not super().keep(read):
            return False
        if self.allowed_read_names is not None:
            if read_key(read.query_name) not in self.allowed_read_names:
                return False
        return True

    def polya_length(self, read: pysam.AlignedSegment) -> int:
        tagged = polya_length_from_tags(read)
        if tagged is not None:
            return max(tagged, 0)
        fallback = self.polya_lengths.get(read_key(read.query_name), 0)
        try:
            return max(int(fallback), 0)
        except (TypeError, ValueError):
            return 0

    def _annotate(self, features: FeatureArray) -> None:
        """Attach poly(A) length and read identity to each feature and frame row."""
        lengths, names = [], []
        for feature in features:
            read = feature.parm["read"]
            length = self.polya_length(read)
            feature.polya_length = length
            feature.read_name = read.query_name
            feature.read_type = "polya" if length >= self.min_polya_length else "untailed"
            feature.transcript_id = ""
            lengths.append(length)
            names.append(read.query_name)

        features.df["polya_length"] = lengths
        features.df["read_name"] = names
        features.df["read_type"] = [
            "polya" if length >= self.min_polya_length else "untailed" for length in lengths
        ]
        if self.draw_polya:
            self._append_tails(features)

    def _append_tails(self, features: FeatureArray) -> None:
        """Extend each read's 3' end by its (scaled) poly(A) tail."""
        for feature in features:
            if feature.polya_length < self.min_polya_length:
                continue
            visual = int(round(feature.polya_length * self.polya_scale))
            if visual <= 0:
                continue
            if feature.strand == "+":
                tail = Feature(feature.end + 1, feature.end + visual,
                               feature.chr_name, feature.strand, "", 1, "polyA")
                feature.children.append(tail)
                feature.end += visual
            else:
                tail = Feature(feature.start - visual, feature.start - 1,
                               feature.chr_name, feature.strand, "", 1, "polyA")
                feature.children.insert(0, tail)
                feature.start -= visual
        # start/end moved, so refresh the frame rows that drive layout.
        features.df["start"] = [f.start for f in features]
        features.df["end"] = [f.end for f in features]

    def open(self) -> None:
        if self.pool is None:
            super().open()

    def close(self) -> None:
        if self.pool is None:
            super().close()

    def iter_region_reads(self):
        if self.pool is None:
            yield from super().iter_region_reads()
            return
        reads = self.pool.reads_for(
            self.chr_name, self.region_start, self.region_end, self.allowed_read_names,
        )
        if self.read_filter is None:
            yield from reads
        else:
            yield from (read for read in reads if self.read_filter(read))

    def _read_region(self) -> FeatureArray:
        features = reads_to_features(self.iter_region_reads())
        self._annotate(features)
        return features

    def max_polya_length(self, region: Sequence) -> int:
        """Longest poly(A) tail among the reads this reader would keep."""
        chr_name, start, end, region_strand = region
        self.chr_name = chr_name
        self.region_start = int(start)
        self.region_end = int(end)
        self.region_strand = region_strand

        self.open()
        try:
            return max(
                (self.polya_length(read) for read in self.iter_region_reads()),
                default=0,
            )
        finally:
            self.close()
