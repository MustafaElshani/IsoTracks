"""IsoTracks: isoform, read and poly(A) figures from IsoForge long-read RNA-seq results.

    from isotracks import plot_gene

    plot_gene("SAT1", isoforge="results/project",
              bam_files={"Control": ["c1.bam"], "Treated": ["t1.bam"]}, output="SAT1")
"""

from .browser import BamInputError, IsoTracks, plot_gene
from .canvas import Canvas
from .fusion_view import FusionView
from .layout import Layout
from .theme import Theme
from .theme import active as active_theme
from .theme import use as use_theme

__version__ = "1.1.0"

__all__ = [
    "FusionView",
    "BamInputError",
    "Canvas",
    "IsoTracks",
    "Layout",
    "Theme",
    "__version__",
    "active_theme",
    "plot_gene",
    "use_theme",
]
