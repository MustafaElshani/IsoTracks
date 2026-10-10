"""Track types."""

from .axis import CoordinateAxisTrack, ScaleBarTrack
from .base import Track
from .coverage import CoverageTrack
from .ends import EndDensityTrack
from .features import FeatureTrack, GeneModelTrack, TranscriptTrack
from .fusion import FusionReadTrack
from .fusion_schematic import FusionSchematicTrack
from .moddiff import ModDiffTrack
from .modifications import ModificationTrack
from .sites import SiteTrack
from .reads import ReadTrack
from .spacer import HighlightLabelTrack, SpacerTrack

#: Name -> class, for configuration files and the ``Canvas.add`` string form.
TRACK_TYPES = {
    cls.__name__: cls
    for cls in (
        CoordinateAxisTrack,
        CoverageTrack,
        EndDensityTrack,
        FeatureTrack,
        FusionReadTrack,
        FusionSchematicTrack,
        GeneModelTrack,
        HighlightLabelTrack,
        ModDiffTrack,
        ModificationTrack,
        SiteTrack,
        ReadTrack,
        ScaleBarTrack,
        SpacerTrack,
        TranscriptTrack,
    )
}

__all__ = [
    "CoordinateAxisTrack",
    "CoverageTrack",
    "EndDensityTrack",
    "FeatureTrack",
    "FusionReadTrack",
    "FusionSchematicTrack",
    "GeneModelTrack",
    "HighlightLabelTrack",
    "ModDiffTrack",
    "ModificationTrack",
    "SiteTrack",
    "ReadTrack",
    "ScaleBarTrack",
    "SpacerTrack",
    "TRACK_TYPES",
    "Track",
    "TranscriptTrack",
]
