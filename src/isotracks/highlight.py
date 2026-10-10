"""Region highlights: one band across every track, marking a stretch of the gene."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy

#: Allowed values of ``highlight_style``.
HIGHLIGHT_STYLES = ("both", "box", "outline")

#: Tracks the band does not run through.
_UNBANDED = ("ScaleBarTrack", "CoordinateAxisTrack", "HighlightLabelTrack")

#: ``start-end`` or ``chrom:start-end``, commas allowed, as a browser prints it.
_COORDINATES = re.compile(
    r"^(?:(?P<chrom>[^:\s]+):)?(?P<start>[\d,]+)\s*[-–]\s*(?P<end>[\d,]+)$"
)

#: Names a region can be asked for by, mapped to the canonical target.
_ALIASES = {
    "3utr": "3_utr", "utr3": "3_utr", "3_prime_utr": "3_utr",
    "three_prime_utr": "3_utr",
    "5utr": "5_utr", "utr5": "5_utr", "5_prime_utr": "5_utr",
    "five_prime_utr": "5_utr",
    "coding": "cds",
}


@dataclass
class Highlight:
    """A region to mark: one or more genomic spans, drawn under one label."""

    spans: List[Tuple[int, int]]
    label: str = ""
    #: Colour of the label.
    label_color: Optional[str] = None
    #: Colour of the band.
    color: Optional[str] = None
    #: The isoform a named region was resolved against, for reporting.
    transcript_id: Optional[str] = None
    #: Overrides the figure's ``highlight_style`` for this region.
    style: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def start(self) -> int:
        return min(start for start, _ in self.spans)

    @property
    def end(self) -> int:
        return max(end for _, end in self.spans)


def parse_coordinates(text: str) -> Optional[Tuple[Optional[str], int, int]]:
    """``(chrom, start, end)`` if *text* is a coordinate range, else None."""
    match = _COORDINATES.match(text.strip())
    if not match:
        return None
    start = int(match["start"].replace(",", ""))
    end = int(match["end"].replace(",", ""))
    if end < start:
        start, end = end, start
    return match["chrom"], start, end


def normalise_target(target: str) -> str:
    """``"3'UTR"``, ``"3-utr"`` and ``"utr3"`` all mean ``"3_utr"``."""
    key = target.strip().lower()
    for mark in ("′", "'"):
        key = key.replace(mark, "_")
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return _ALIASES.get(key.replace("_", ""), _ALIASES.get(key, key))


def split_spec(spec: str) -> Tuple[str, Optional[str]]:
    """``"exon_2:ENST0001"`` -> ``("exon_2", "ENST0001")``."""
    if parse_coordinates(spec) is not None:
        return spec, None
    target, _, transcript = spec.partition(":")
    return target.strip(), (transcript.strip() or None)


def draw_highlights(canvas, highlights: Sequence[Highlight], style: str,
                    theme) -> List[Rectangle]:
    """Draw *highlights* across the rendered *canvas*; return the patches."""
    if style not in HIGHLIGHT_STYLES:
        raise ValueError(
            f"unknown highlight style {style!r}; expected one of "
            + ", ".join(repr(s) for s in HIGHLIGHT_STYLES)
        )
    if not highlights or canvas.fig is None:
        return []

    def axes_in(index):
        def pick(track):
            if index == 0:
                return track.ax
            copies = getattr(track, "panel_copies", None) or []
            return copies[index - 1].ax if len(copies) >= index else None
        return pick

    panels = [(panel.region, axes_in(index))
              for index, panel in enumerate(getattr(canvas, "panels", [None]))
              if panel is not None] or [(canvas.region, axes_in(0))]

    labelled_above = any(type(t).__name__ == "HighlightLabelTrack"
                         for t in canvas.tracks)
    drawn: List[Rectangle] = []
    for region, axes_of in panels:
        banded = [axes_of(t) for t in canvas.tracks
                  if type(t).__name__ not in _UNBANDED and not t.SPANS_PANELS
                  and axes_of(t) is not None]
        if banded:
            drawn += _band_panel(canvas.fig, banded, region, highlights, style,
                                 theme, labelled_above)
    return drawn


def _band_panel(figure, axes, region, highlights, style, theme,
                labelled_above) -> List[Rectangle]:
    """Band one panel: *axes* are its data axes, top to bottom."""
    shared = axes[0]
    top = max(ax.get_position().y1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    transform = blended_transform_factory(shared.transData, figure.transFigure)

    low, high = sorted(region[1:3])
    inverted = region[3] == "-"
    edge_width = 0.7 * theme.stroke
    drawn: List[Rectangle] = []

    for highlight in highlights:
        color = highlight.color or theme.highlight_color
        own_style = highlight.style or style
        visible = []
        for start, end in highlight.spans:
            left = max(float(start) - 0.5, float(low))
            right = min(float(end) + 0.5, float(high))
            if right <= left:
                continue
            visible.append((left, right))
            if own_style in ("box", "both"):
                drawn.append(figure.add_artist(Rectangle(
                    (left, bottom), right - left, top - bottom,
                    transform=transform, facecolor=color, edgecolor="none",
                    alpha=theme.highlight_alpha, zorder=4,
                )))
            if own_style in ("outline", "both"):
                drawn.append(figure.add_artist(Rectangle(
                    (left, bottom), right - left, top - bottom,
                    transform=transform, fill=False, edgecolor=color,
                    linewidth=edge_width, linestyle=(0, (1.2, 1.6)), zorder=5,
                )))
        if visible and highlight.label and not labelled_above:
            anchor = (max(r for _, r in visible) if inverted
                      else min(l for l, _ in visible))
            text = figure.text(
                anchor, top, highlight.label,
                transform=offset_copy(transform, fig=figure,
                                      x=2.0, y=-1.5, units="points"),
                ha="left", va="top", zorder=6,
                color=highlight.label_color or theme.ink["muted"],
                fontsize=theme.fonts["annot"] * 0.9,
            )
            _halo(text, theme)
    return drawn


def _halo(text, theme) -> None:
    if theme.transparent:
        return
    text.set_path_effects([
        patheffects.withStroke(linewidth=2.2, foreground="white"),
        patheffects.Normal(),
    ])
