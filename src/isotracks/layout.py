"""How tall each track is, and therefore how tall the figure is."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .theme import MM_TO_IN
from .theme import active as active_theme

#: Printed height of one layout unit.
MM_PER_UNIT = 4.2

#: Space above and below the track stack (scale bar, axis caption, breathing room).
FRAME_MM = 10.0

#: Tallest figure, in millimetres.
MAX_HEIGHT_MM = 900.0


#: Resolutions text is measured at; font hinting changes its width with each.
MEASURE_DPI = (72, 96, 100, 150, 200, 220, 300, 600)
#: Height of one line of type in ems, descenders included.
LINE_BOX_EM = 1.25
#: Right edge of gutter labels as a share of the column, and the space kept clear to their left, in
#: inches.
GUTTER_TEXT_X = 0.955
GUTTER_PAD_IN = 0.04


@dataclass(frozen=True)
class Layout:
    """Track heights in layout units.

        IsoTracks(..., layout=Layout(read_stack=4.5))
    """

    #: Annotation tracks above the models.
    tss_sites: float = 0.85
    polya_sites: float = 0.85
    #: Read 5' and 3' end density.
    end_density: float = 1.1
    #: The gene model: the drawn isoforms overlaid on one row.
    gene_model: float = 0.95
    #: One isoform model.
    transcript: float = 0.80
    #: One read stack.
    read_stack: float = 3.0
    #: RNA modification lollipops, one per condition.
    modification: float = 2.4
    #: Differential modification, one per comparison.
    modification_diff: float = 2.6
    #: Read depth with junction arcs, one per condition.
    coverage: float = 3.0
    #: The isoform heading above its stacks, and the gap before the next group.
    group_heading: float = 0.62
    group_gap: float = 0.30
    #: The fusion schematic: partner genes, where they join, and the fused transcript.
    fusion_schematic: float = 3.2
    #: Scale bar and coordinate axis.
    scale_bar: float = 0.65
    axis: float = 1.25

    @classmethod
    def compact(cls, **overrides) -> "Layout":
        """A tighter stack, for a figure that has to share a page."""
        return cls(**{**COMPACT_HEIGHTS, **overrides})

    def figure_size(self, total_units: float, width=None) -> Tuple[float, float]:
        """Figure size in inches for a stack totalling *total_units*."""
        theme = active_theme()
        width_mm = theme.width_mm if width is None else theme.derive(width=width).width_mm
        height_mm = min(FRAME_MM + MM_PER_UNIT * max(total_units, 1.0), MAX_HEIGHT_MM)
        return (width_mm * MM_TO_IN, height_mm * MM_TO_IN)

    def label_units(self, lines: int = 2) -> float:
        """Units a gutter label of *lines* lines needs, at the active type size."""
        return label_block_units(lines)

    def gutter_width(self, longest_label: int) -> float:
        """Label-column width in inches for the longest label to be shown."""
        theme = active_theme()
        char_in = theme.fonts["gutter"] * 0.55 / 72.0
        return float(min(max(0.42, longest_label * char_in + 0.08), 1.5))

    def gutter_width_for(self, labels: Sequence[str], bold: Sequence[str] = ()) -> float:
        """Label-column width in inches for these labels, measured in the active font."""
        size = active_theme().fonts["gutter"]
        widest = max([text_width_in(text, size) for text in labels if text]
                     + [text_width_in(text, size, "bold") for text in bold if text] + [0.0])
        return float(min(max(0.42, (widest + GUTTER_PAD_IN) / GUTTER_TEXT_X), 1.5))


def text_width_in(text: str, size: float, weight: str = "normal") -> float:
    """Widest *text* is drawn, in inches, over the resolutions in :data:`MEASURE_DPI`."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(size=size, weight=weight)
    return max(RendererAgg(1, 1, dpi).get_text_width_height_descent(text, prop, False)[0] / dpi
               for dpi in MEASURE_DPI)


def type_units(points: float, lines: int = 1) -> float:
    """Height in layout units of *lines* lines set at *points*."""
    total = points * (1.25 * (lines - 1) + 1.0)
    return (total / 72.0 * 25.4) / MM_PER_UNIT


def label_block_units(lines: int = 2) -> float:
    """Units a gutter label of *lines* lines needs, at the active type size."""
    if lines < 2:
        return 0.0
    return type_units(active_theme().fonts["gutter"], lines)


#: Heights :meth:`Layout.compact` overrides.
COMPACT_HEIGHTS = {
    "end_density": 0.80,
    "gene_model": 0.85,
    "transcript": 0.68,
    "read_stack": 1.80,
    "modification": 1.60,
    "modification_diff": 1.70,
    "coverage": 1.90,
    "group_heading": 0.52,
    "group_gap": 0.18,
}

#: Longest gutter label kept intact; anything longer is elided in the middle.
MAX_LABEL_CHARS = 20


def shorten_label(text: str, limit: int = MAX_LABEL_CHARS) -> str:
    """Elide the middle of an over-long label, keeping both ends readable."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}…{text[len(text) - tail:]}"
