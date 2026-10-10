"""Whitespace and section headings."""

from __future__ import annotations

from typing import Any, Dict

from ..layout import type_units
from .base import Track


class SpacerTrack(Track):
    """Blank row, optionally carrying a heading and a separating rule."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 0.4,
        "heading": None,  # drawn inside the axes, left-aligned
        "heading_color": None,  # None -> theme text ink
        # A shade smaller than the annotation size.
        "heading_scale": 0.88,
        "heading_gap_pt": 2.0,  # clearance below the rule, in points
        # "all" repeats the heading in every panel; "first" draws it once.
        "heading_panels": "all",
        # Drawn instead of the heading when the heading would run past the figure's right edge.
        "heading_short": None,
    }

    def heading_size(self) -> float:
        return self.theme.fonts["annot"] * float(self.config["heading_scale"])

    def min_height(self) -> float:
        """Enough for the heading plus the gap that keeps it off the rule."""
        floor = super().min_height()
        if not self.config["heading"]:
            return floor
        gap = type_units(float(self.config["heading_gap_pt"]))
        return max(floor, type_units(self.heading_size()) + gap)

    def draw(self) -> None:
        self.set_ylim(0, 1)
        heading = self.config["heading"]
        if not heading:
            return
        if self.config["heading_panels"] == "first" and self.panel != 0:
            return
        theme = self.theme
        text = self.ax.annotate(
            heading, xy=(0.0, 1.0), xycoords=self.ax.transAxes,
            xytext=(0, -float(self.config["heading_gap_pt"])),
            textcoords="offset points",
            ha="left", va="top",
            color=self.config["heading_color"] or theme.ink["text"],
            fontsize=self.heading_size(), fontweight="bold",
            annotation_clip=False,
        )
        short = self.config["heading_short"]
        if short and short != heading:
            figure = self.ax.figure
            extent = text.get_window_extent(figure.canvas.get_renderer())
            if extent.x1 > figure.bbox.x1:
                text.set_text(short)


class HighlightLabelTrack(Track):
    """A row above the tracks naming each highlighted region, centred over it."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 0.55,
        "highlights": [],  # Highlight objects; only their labels are drawn
        "label_scale": 0.9,  # of the annotation size
        "label_gap_pt": 1.5,  # clearance above the band, in points
    }

    def label_size(self) -> float:
        return self.theme.fonts["annot"] * float(self.config["label_scale"])

    def min_height(self) -> float:
        return max(super().min_height(),
                   type_units(self.label_size())
                   + type_units(float(self.config["label_gap_pt"])) * 2)

    def draw(self) -> None:
        self.set_ylim(0, 1)
        low, high = sorted((float(self.region_start), float(self.region_end)))
        theme = self.theme
        for highlight in self.config["highlights"]:
            if not highlight.label:
                continue
            visible = [(max(s - 0.5, low), min(e + 0.5, high))
                       for s, e in highlight.spans
                       if min(e + 0.5, high) > max(s - 0.5, low)]
            if not visible:
                continue
            to_drawn, to_genomic = self._drawn_units()
            ends = [float(to_drawn(v)) for pair in visible for v in pair]
            centre = float(to_genomic((min(ends) + max(ends)) / 2.0))
            text = self.ax.annotate(
                highlight.label, xy=(centre, 0.0),
                xytext=(0, float(self.config["label_gap_pt"])),
                textcoords="offset points", ha="center", va="bottom",
                color=highlight.label_color or theme.ink["muted"],
                fontsize=self.label_size(), annotation_clip=False,
            )
            self.halo(text)
