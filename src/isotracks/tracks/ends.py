"""Read-end density: the evidence under a TSS or poly(A) call."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..io.ends import ReadEndReader
from .base import Track


class EndDensityTrack(Track):
    """Smoothed density of read 5' and 3' ends."""

    #: A detail panel keeps this panel's scale.
    PANEL_SHARED_Y = True

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": ReadEndReader,
        "height": 2.2,
        "show_scale": False,  # reported in the gutter
        "label_lines": 2,  # the gutter label takes two lines
        "y_margin": (0.0, 0.18),
        "five_prime_color": None,  # None -> theme TSS colour
        "three_prime_color": None,  # None -> theme poly(A) signal colour
        "fill_alpha": 0.5,
        "line_width": 0.7,
        # Off by default.
        "label_curves": False,
        "five_prime_label": "5′ ends",
        "three_prime_label": "3′ ends",
    }

    def draw(self) -> None:
        density = self.region_data
        fixed = self.config["fix_ylim"]
        if density is None or density.is_empty():
            self.set_ylim(0, 1)
            self.config["sublabel"] = (f"0\u2013{self._format_scale(fixed[1])}"
                                       if fixed else "")
            return

        theme = self.theme
        self.set_ylim(0.0, max(density.peak, 1e-9))
        top = fixed[1] if fixed else density.peak
        self.config["sublabel"] = f"0\u2013{self._format_scale(top)}"

        curves = (
            (density.five_prime, density.five_prime_reads,
             self.config["five_prime_color"] or theme.tss_color,
             self.config["five_prime_label"]),
            (density.three_prime, density.three_prime_reads,
             self.config["three_prime_color"] or theme.polya_signal_color,
             self.config["three_prime_label"]),
        )
        for values, reads, color, label in curves:
            if reads == 0:
                continue
            self._fill(density.positions, values, color)
            if self.config["label_curves"]:
                self._label_peak(density.positions, values, color, label)

        self.ax.axhline(0, color=theme.ink["hairline"],
                        linewidth=0.5 * theme.stroke, zorder=3)

    def _fill(self, positions: np.ndarray, values: np.ndarray, color: str) -> None:
        stroke = self.theme.stroke
        self.ax.fill_between(
            positions, 0.0, values, facecolor=color,
            alpha=self.config["fill_alpha"], linewidth=0, zorder=5,
        )
        self.ax.plot(
            positions, values, color=color,
            linewidth=self.config["line_width"] * stroke, zorder=6,
        )

    def _label_peak(self, positions: np.ndarray, values: np.ndarray,
                    color: str, label: str) -> None:
        """Name a curve at its own peak, so no legend is needed."""
        if not len(values):
            return
        index = int(np.argmax(values))
        if values[index] <= 0:
            return

        where = index / max(len(values) - 1, 1)
        if self.x_inverted:
            where = 1.0 - where
        align = "left" if where < 0.12 else "right" if where > 0.88 else "center"
        offset = {"left": 2, "right": -2, "center": 0}[align]

        text = self.ax.annotate(
            label, xy=(positions[index], values[index]),
            xytext=(offset, 2), textcoords="offset points",
            ha=align, va="bottom", color=color,
            fontsize=self.theme.fonts["annot"], zorder=8,
        )
        self.halo(text)

    @staticmethod
    def _format_scale(value: float) -> str:
        """The scale as whole reads per 100 bp."""
        if value >= 10:
            return f"{value:,.0f}"
        return f"{value:.1f}".rstrip("0").rstrip(".")
