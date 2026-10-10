"""Genomic features and the BED12 reader."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURE_HEADER: List[str] = ["chr_name", "start", "end", "strand", "height", "id", "type"]


class Feature:
    """One span, 1-based inclusive, optionally with children.

    Attributes
    ----------
    chr_name, start, end, strand : basic placement
    height : vertical units the feature occupies when stacked
    id : identifier, empty for child features
    type : ``"mRNA"``, ``"exon"``, ``"intron"``, ``"CDS"``, ``"5UTR"``,
        ``"3UTR"`` or ``"polyA"``
    parm : per-feature style overrides consulted before the track config
    children : nested features drawn inside this one's row
    """

    __slots__ = (
        "chr_name", "start", "end", "strand", "height", "id", "type",
        "parm", "children", "y_min", "y_max",
        "polya_length", "read_name", "read_type", "transcript_id",
    )

    def __init__(
        self,
        start: int,
        end: int,
        chr_name: Optional[str] = None,
        strand: Optional[str] = None,
        feature_id: str = "",
        height: float = 1,
        type_name: Optional[str] = None,
        parm: Optional[Dict[str, Any]] = None,
        children: Optional[List["Feature"]] = None,
    ):
        self.chr_name = chr_name
        self.start = start
        self.end = end
        self.strand = strand
        self.height = height
        self.id = feature_id
        self.type = type_name
        self.parm = dict(parm) if parm else {}
        self.children = list(children) if children else []
        self.y_min = 0.0
        self.y_max = 0.0
        self.polya_length = 0
        self.read_name = ""
        self.read_type = ""
        self.transcript_id = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Feature({self.type} {self.chr_name}:{self.start}-{self.end}{self.strand})"


class FeatureArray:
    """A list of :class:`Feature` objects with a parallel DataFrame view."""

    def __init__(self, features: Optional[Sequence[Feature]] = None, header: Optional[List[str]] = None):
        self.header = list(header) if header else list(FEATURE_HEADER)
        self.features: List[Feature] = list(features) if features else []
        self.df = self._build_frame()
        self.index: Dict[str, Feature] = {}

    def _build_frame(self) -> pd.DataFrame:
        rows = [[getattr(f, name) for name in self.header] for f in self.features]
        df = pd.DataFrame(rows, columns=self.header)
        df["index_order"] = np.arange(len(df))
        return df

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def is_empty(self) -> bool:
        return len(self.df) == 0

    def build_index(self, include_children: bool = False) -> None:
        """Index features by id so :meth:`get` can find them."""
        index: Dict[str, Feature] = {}
        for feature in self.features:
            if feature.id:
                index[feature.id] = feature
            if include_children:
                for child in feature.children:
                    if child.id:
                        index[child.id] = child
        self.index = index

    def get(self, feature_id: str) -> Feature:
        if not self.index:
            self.build_index()
        return self.index[feature_id]

    def push_columns(self, columns: Sequence[str]) -> None:
        """Copy DataFrame columns back onto the feature objects."""
        for column in columns:
            for position, value in zip(self.df["index_order"], self.df[column]):
                setattr(self.features[position], column, value)

    def reorder_to_frame(self) -> None:
        """Reorder ``features`` to match the current row order of ``df``."""
        self.features = [self.features[i] for i in self.df["index_order"]]
        self.df = self.df.copy()
        self.df["index_order"] = np.arange(len(self.df))

    def subset(self, positions: Sequence[int]) -> "FeatureArray":
        """A new array of the same class holding only the given rows."""
        return type(self)(features=[self.features[i] for i in positions], header=self.header)

    # ------------------------------------------------------------- reading

    def read(self, file_name: str) -> List[Feature]:  # pragma: no cover - overridden
        raise NotImplementedError

    def load_region_data(self, region: Sequence, select_features: Optional[Sequence[str]] = None) -> None:
        """Select the features overlapping *region* into ``self.region_data``."""
        chr_name, region_start, region_end, _strand = region
        if select_features:
            self.build_index()
            self.region_data = type(self)(
                features=[self.index[fid] for fid in select_features if fid in self.index],
                header=self.header,
            )
            return

        df = self.df
        on_chrom = df["chr_name"] == chr_name
        overlaps = (df["start"] <= region_end) & (df["end"] >= region_start)
        self.region_data = self.subset(df["index_order"][on_chrom & overlaps])

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


class TranscriptFeatures(FeatureArray):
    """Transcript models, with introns filled in between exon-like children."""

    def __init__(self, features=None, file_name: Optional[str] = None, header=None):
        if features is None and file_name is not None:
            self.header = list(header) if header else list(FEATURE_HEADER)
            features = self.read(file_name)
        super().__init__(features, header)

    @staticmethod
    def add_introns(features: Sequence[Feature]) -> List[Feature]:
        """Insert intron children between consecutive exon-like children."""
        for feature in features:
            children = feature.children
            if not children:
                continue
            children.sort(key=lambda f: f.start)
            if len(children) == 1:
                children[0].parm["is_single"] = True
            elif feature.strand == "+":
                children[0].parm["is_first"] = True
                children[-1].parm["is_last"] = True
            else:
                children[-1].parm["is_first"] = True
                children[0].parm["is_last"] = True

            introns = []
            for left, right in zip(children[:-1], children[1:]):
                if right.start > left.end + 1:
                    introns.append(Feature(
                        left.end + 1, right.start - 1, feature.chr_name,
                        feature.strand, "", feature.height, "intron",
                    ))
            children.extend(introns)
        return list(features)


class BedTranscripts(TranscriptFeatures):
    """BED12 (or BED6/BED3) transcript models."""

    def read(self, file_name: str, based0: bool = True, region_type_name: str = "mRNA") -> List[Feature]:
        features: List[Feature] = []
        with open(file_name) as handle:
            for line in handle:
                if line.startswith(("browser", "track", "#")) or not line.strip():
                    continue
                fields = line.rstrip("\n").split()
                feature = self._parse_line(fields, based0, region_type_name)
                if feature is not None:
                    features.append(feature)
        return self.add_introns(features)

    def _parse_line(self, fields: List[str], based0: bool, region_type_name: str) -> Optional[Feature]:
        if len(fields) < 3:
            return None
        chr_name = fields[0]
        region_start, region_end = int(fields[1]), int(fields[2])
        region_id = fields[3] if len(fields) > 3 else f"{chr_name}:{region_start}:{region_end}"
        strand = fields[5] if len(fields) > 5 else "+"

        if len(fields) >= 12:
            children = self._blocks_to_children(fields, chr_name, strand, region_start, based0)
        else:
            start = region_start + 1 if based0 else region_start
            children = [Feature(start, region_end, chr_name, strand, "", 1, "CDS")]

        if based0:
            region_start += 1
        return Feature(region_start, region_end, chr_name, strand, region_id,
                       1, region_type_name, children=children)

    @staticmethod
    def _blocks_to_children(fields, chr_name, strand, region_start, based0) -> List[Feature]:
        sizes = [int(v) for v in fields[10].rstrip(",").split(",") if v != ""]
        offsets = [int(v) for v in fields[11].rstrip(",").split(",") if v != ""]
        blocks = [(region_start + off, region_start + off + size) for off, size in zip(offsets, sizes)]
        thick_start, thick_end = int(fields[6]), int(fields[7])

        # Non-coding: render every block as thin UTR.
        if thick_start >= thick_end:
            return [
                Feature(bs + 1 if based0 else bs, be, chr_name, strand, "", 1, "5UTR")
                for bs, be in blocks
            ]

        left_utr: List[Tuple[int, int]] = []
        cds: List[Tuple[int, int]] = []
        right_utr: List[Tuple[int, int]] = []
        for block_start, block_end in blocks:
            if block_end <= thick_start:
                left_utr.append((block_start, block_end))
                continue
            if block_start >= thick_end:
                right_utr.append((block_start, block_end))
                continue
            if block_start < thick_start:
                left_utr.append((block_start, thick_start))
            cds.append((max(block_start, thick_start), min(block_end, thick_end)))
            if block_end > thick_end:
                right_utr.append((thick_end, block_end))

        names = ("3UTR", "CDS", "5UTR") if strand == "-" else ("5UTR", "CDS", "3UTR")
        children: List[Feature] = []
        for spans, type_name in zip((left_utr, cds, right_utr), names):
            for start, end in spans:
                if end <= start:
                    continue
                children.append(Feature(start + 1 if based0 else start, end,
                                        chr_name, strand, "", 1, type_name))
        return children
