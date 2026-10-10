"""Coordinate axis and scale bar."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from matplotlib import ticker

from ..layout import LINE_BOX_EM, type_units
from .base import Track

#: Round scale-bar lengths, in base pairs.
SCALE_STEPS = np.array([
    10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000, 1_000_000,
])


def format_coordinate(value: float, span: float,
                      ticks: Sequence[float] = ()) -> str:
    """A coordinate label whose unit suits the width of the view."""
    if span >= 1e6:
        return f"{value / 1e6:,.2f} Mb"
    whole_kb = bool(ticks) and all(float(t) % 1_000 == 0 for t in ticks)
    if whole_kb:
        return f"{value / 1e3:,.0f} kb"
    if span >= 1e4:
        return f"{value / 1e3:,.1f} kb"
    return f"{value:,.0f}"


def format_length(bp: float) -> str:
    """A scale-bar length such as ``500 bp``, ``2 kb``, ``1 Mb``."""
    if bp >= 1e6:
        return f"{bp / 1e6:g} Mb"
    if bp >= 1e3:
        return f"{bp / 1e3:g} kb"
    return f"{bp:g} bp"


class CoordinateAxisTrack(Track):
    """Genomic coordinate axis: one hairline, a few ticks, one caption."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 1.2,
        "ticks": 5,
        "tick_length": 0.16,  # fraction of the track height
        "axis_y": 0.86,  # where the axis line sits within the track
        "show_caption": True,
        "caption": None,  # None -> "GENE  ·  chrN   + strand"
        "gene": "",  # named ahead of the locus when set
        "color": None,  # None -> theme axis ink
    }

    def tick_positions(self) -> List[float]:
        """Round tick values, thinned to however many labels actually fit."""
        low, high = sorted((float(self.region_start), float(self.region_end)))
        span = high - low
        wanted = min(self.config["ticks"], self._labels_that_fit(low, high, span))
        if self.intron_map is not None:
            return self._shortened_ticks(low, high, wanted)
        locator = ticker.MaxNLocator(
            nbins=max(wanted, 2), steps=[1, 2, 2.5, 5, 10], min_n_ticks=2,
        )
        ticks = [t for t in locator.tick_values(low, high) if low <= t <= high]
        return ticks or [low, high]

    def _shortened_ticks(self, low: float, high: float, wanted: int) -> List[float]:
        """Ticks spread evenly across the *drawn* axis, at tidy genomic values."""
        to_drawn, to_genomic = self._drawn_units()
        near, far = float(to_drawn(low)), float(to_drawn(high))
        count = max(wanted, 2)
        # Inset by half a step.
        step = (far - near) / count
        drawn = [near + step * (index + 0.5) for index in range(count)]
        unit = 1_000.0 if high - low >= 5_000 else 100.0
        ticks = sorted({round(float(to_genomic(d)) / unit) * unit for d in drawn})
        return [t for t in ticks if low <= t <= high] or [low, high]

    def _labels_that_fit(self, low: float, high: float, span: float) -> int:
        axes_width_in = self.ax.get_window_extent().width / self.ax.figure.dpi
        if axes_width_in <= 0:
            return self.config["ticks"]
        sample = (1_000.0,) if span >= 5_000 else ()
        widest = max(len(format_coordinate(value, span, sample)) for value in (low, high))
        # 0.6 em per character, plus half a label of clear space.
        label_in = widest * 0.6 * self.theme.fonts["tick"] / 72.0
        return max(int(axes_width_in // (label_in * 1.5)), 2)

    def draw(self) -> None:
        self.set_ylim(0, 1)
        theme = self.theme
        color = self.config["color"] or theme.ink["axis"]
        stroke = theme.stroke
        axis_y = self.config["axis_y"]
        tick_bottom = axis_y - self.config["tick_length"]

        ticks = self.tick_positions()
        low, high = sorted((float(self.region_start), float(self.region_end)))
        span = high - low

        self.ax.plot([low, high], [axis_y, axis_y], color=color,
                     linewidth=0.6 * stroke, solid_capstyle="butt", zorder=5)
        self._draw_breaks(axis_y, color)
        marks = []
        for position in ticks:
            (tick,) = self.ax.plot([position, position], [tick_bottom, axis_y],
                                   color=color, linewidth=0.6 * stroke, zorder=5)
            label = self.ax.text(
                position, tick_bottom - 0.08, format_coordinate(position, span, ticks),
                ha="center", va="top", color=color, fontsize=theme.fonts["tick"],
            )
            marks.append((tick, label))
        self._drop_colliding_labels(marks)

        caption_here = (self.panel == 0 if self.panel_spec is None
                        else self.panel_spec.caption)
        if self.config["show_caption"] and caption_here:
            self.ax.text(
                0.5, -0.04, self._caption(), transform=self.ax.transAxes,
                ha="center", va="top", color=theme.ink["muted"],
                fontsize=theme.fonts["annot"],
            )

    def _draw_breaks(self, axis_y: float, color: str) -> None:
        """A break mark, ``//``, across the axis line at every shortened intron."""
        if self.intron_map is None or not self.intron_map.breaks:
            return
        to_drawn, to_genomic = self._drawn_units()
        figure = self.ax.figure
        axes = self.ax.get_window_extent()
        low, high = sorted((float(self.region_start), float(self.region_end)))
        drawn_span = float(to_drawn(high) - to_drawn(low)) or 1.0
        # Points to drawn units horizontally, and to the axes' 0-1 vertically.
        per_point_x = drawn_span / max(axes.width, 1.0) * figure.dpi / 72.0
        per_point_y = (self.ylim[1] - self.ylim[0]) / max(axes.height, 1.0) * figure.dpi / 72.0
        gap = 1.0 * per_point_x  # half the cut in the line
        lean = 1.1 * per_point_x  # how far each stroke leans
        rise = 2.6 * per_point_y  # half each stroke's height
        page = "none" if self.theme.transparent else "white"
        stroke = 0.6 * self.theme.stroke

        for start, end in self.intron_map.breaks:
            middle = (float(to_drawn(start - 0.5)) + float(to_drawn(end + 0.5))) / 2.0
            if not (to_drawn(low) < middle < to_drawn(high)):
                continue
            # Cut the line, then draw a stroke at each edge of the cut.
            self.ax.fill_between(
                [float(to_genomic(middle - gap)), float(to_genomic(middle + gap))],
                axis_y - rise * 0.4, axis_y + rise * 0.4,
                color=page, linewidth=0, zorder=6,
            )
            for edge in (middle - gap, middle + gap):
                xs = [float(to_genomic(edge - lean)), float(to_genomic(edge + lean))]
                self.ax.plot(xs, [axis_y - rise, axis_y + rise], color=color,
                             linewidth=stroke, solid_capstyle="butt", zorder=7,
                             gid="intron-break")

    def _drop_colliding_labels(self, marks) -> None:
        """Remove any tick whose label would touch its neighbour's."""
        if len(marks) < 2 and not self._beside_another_panel():
            return
        figure = self.ax.figure
        renderer = figure.canvas.get_renderer()
        gap = 0.5 * self.theme.fonts["tick"] / 72.0 * figure.dpi
        left_limit, right_limit = self._label_limits(renderer)

        measured = sorted(((label.get_window_extent(renderer), tick, label)
                           for tick, label in marks), key=lambda m: m[0].x0)
        kept_right = None
        for box, tick, label in measured:
            fits = box.x0 >= left_limit and box.x1 <= right_limit
            clear = kept_right is None or box.x0 >= kept_right + gap
            if fits and clear:
                kept_right = box.x1
            else:
                tick.remove()
                label.remove()

    def _beside_another_panel(self) -> bool:
        return self.panel_count > 1

    def _label_limits(self, renderer) -> Tuple[float, float]:
        """How far, in pixels, this panel's labels may reach either way."""
        if not self._beside_another_panel():
            return float("-inf"), float("inf")
        box = self.ax.get_window_extent(renderer)
        half_gap = self.canvas.panel_gap_in * self.ax.figure.dpi / 2.0
        left = float("-inf") if self.panel == 0 else box.x0 - half_gap
        right = float("inf") if self.panel == self.panel_count - 1 else box.x1 + half_gap
        return left, right

    def _caption(self) -> str:
        caption = self.config["caption"]
        if caption is not None:
            return caption
        strand = self.region_strand or "+"
        locus = f"{self.chr_name}   {strand} strand"
        if self.intron_map is not None:
            # Note that introns are shortened.
            locus += "   \u00b7   introns shortened"
        gene = self.config["gene"]
        return f"{gene}   \u00b7   {locus}" if gene else locus


class ScaleBarTrack(Track):
    """A round-numbered scale bar, right-aligned, with the length beneath it."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 0.7,
        "target_fraction": 0.18,  # of the view the bar should span
        "right_inset": 0.02,  # of the view, kept clear at the right edge
        "bar_y": 0.80,  # high in the row; the label hangs below
        "tick_half": 0.15,  # tick reach either side of the bar line
        "label_gap_pt": 1.5,  # clearance below the bar, in points
        "color": None,  # None -> theme muted ink
    }

    def min_height(self) -> float:
        """The bar, and one line of type hanging below it."""
        margin_low, margin_high = self.config["y_margin"]
        span = 1.0 + float(margin_low) + float(margin_high)
        below = float(self.config["bar_y"]) + float(margin_low)
        if below <= 0:
            return super().min_height()
        text = (type_units(self.theme.fonts["annot"] * LINE_BOX_EM)
                + type_units(float(self.config["label_gap_pt"])))
        return max(super().min_height(), text * span / below)

    def choose_length(self, span: float) -> float:
        """Largest round length at or below the target share of the view."""
        target = span * self.config["target_fraction"]
        candidates = SCALE_STEPS[SCALE_STEPS <= max(target, SCALE_STEPS[0])]
        return float(candidates[-1]) if len(candidates) else float(SCALE_STEPS[0])

    def draw(self) -> None:
        self.set_ylim(0, 1)
        if self.panel_spec is not None and not self.panel_spec.scale_bar:
            return  # the same scale is drawn in a neighbour
        theme = self.theme
        color = self.config["color"] or theme.ink["muted"]

        low, high = sorted((float(self.region_start), float(self.region_end)))
        to_drawn, to_genomic = self._drawn_units()
        near, far = float(to_drawn(low)), float(to_drawn(high))
        span = far - near
        length = self.choose_length(span)
        inset = span * self.config["right_inset"]

        if self.x_inverted:
            drawn_start = near + inset
            drawn_end = drawn_start + length
        else:
            drawn_end = far - inset
            drawn_start = drawn_end - length
        bar_start, bar_end = float(to_genomic(drawn_start)), float(to_genomic(drawn_end))
        middle = float(to_genomic((drawn_start + drawn_end) / 2.0))

        y = self.config["bar_y"]
        reach = self.config["tick_half"]
        self.ax.plot([bar_start, bar_end], [y, y], color=color,
                     linewidth=0.8 * theme.stroke, solid_capstyle="butt", zorder=5)
        for x in (bar_start, bar_end):
            self.ax.plot([x, x], [y - reach, y + reach], color=color,
                         linewidth=0.8 * theme.stroke, zorder=5)

        self.ax.annotate(
            format_length(length), xy=(middle, y),
            xytext=(0, -float(self.config["label_gap_pt"])),
            textcoords="offset points", ha="center", va="top",
            color=color, fontsize=theme.fonts["annot"], annotation_clip=False,
        )
