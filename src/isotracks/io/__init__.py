"""Readers that turn files on disk into things IsoTracks can draw."""

from .alignments import (
    CoverageReader,
    read_blocks,
    read_introns,
    reads_to_features,
)
from .clusters import ClusterSiteReader
from .ends import EndDensity, ReadEndReader
from .moddiff import DiffSites, ModDiffReader
from .modifications import ModPileupReader, ModSites, mod_label
from .features import (
    BedTranscripts,
    Feature,
    FeatureArray,
    TranscriptFeatures,
)
from .longreads import (
    POLYA_VISUAL_SCALE,
    LongReadBam,
    RegionReadPool,
    polya_length_from_tags,
    read_key,
)

__all__ = [
    "BedTranscripts",
    "ClusterSiteReader",
    "CoverageReader",
    "DiffSites",
    "EndDensity",
    "Feature",
    "FeatureArray",
    "LongReadBam",
    "ModDiffReader",
    "ModPileupReader",
    "ModSites",
    "POLYA_VISUAL_SCALE",
    "ReadEndReader",
    "RegionReadPool",
    "TranscriptFeatures",
    "polya_length_from_tags",
    "read_blocks",
    "read_introns",
    "mod_label",
    "read_key",
    "reads_to_features",
]
