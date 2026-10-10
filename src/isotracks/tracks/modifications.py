"""RNA modification track: a lollipop per called site."""

from __future__ import annotations

from typing import Any, Dict, List

from matplotlib.collections import LineCollection

from ..io.modifications import ModPileupReader, mod_label
from .base import Track


class ModificationTrack(Track):
    """Modified fraction at each called site, as a lollipop."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": ModPileupReader,
        "height": 2.4,
        "label_lines": 2,  # the site count is written during draw()
        "show_scale": False,  # the axis is always 0-100%
        "y_margin": (0.0, 0.14),
        "colors": {},  # code -> colour; filled in by the browser
        "stem_width": 0.7,
        "dot_size": 3.2,
        "baseline": True,
        "half_line": True,  # a hairline at 50%, to read heights against
        "show_key": False,  # only the topmost modification track needs one
        "key_gap_em": 1.6,  # clearance between key entries, in ems
    }

    def draw(self) -> None:
        sites = self.region_data
        self.set_ylim(0.0, 1.0)  # the axis is never rescaled

        theme = self.theme
        if self.config["baseline"]:
            self.ax.axhline(0, color=theme.ink["hairline"],
                            linewidth=0.5 * theme.stroke, zorder=3)
        if self.config["half_line"]:
            self.ax.axhline(0.5, color=theme.ink["hairline"],
                            linewidth=0.5 * theme.stroke, linestyle=(0, (2, 2)),
                            zorder=3)

        assessed = getattr(sites, "assessed", 0) if sites is not None else 0
        if sites is None or sites.is_empty():
            self.config["sublabel"] = self._site_count(0, assessed)
            return

        for code in sites.codes():
            self._draw_code(sites.for_code(code), self._color(code))

        self.config["sublabel"] = self._site_count(len(sites.frame), assessed)
        if self.config["show_key"] and self.panel == 0:
            self._draw_key(self._key_codes(sites.codes()))

    def _key_codes(self, present: List[str]) -> List[str]:
        """Codes to name in the key: the ones asked for, in that order."""
        wanted = getattr(self.reader, "codes", None)
        if not wanted:
            return present
        return [code for code in wanted if code in present] or list(wanted)

    def _color(self, code: str) -> str:
        return self.config["colors"].get(str(code), self.theme.ink["muted"])

    def _draw_code(self, frame, color: str) -> None:
        stroke = self.theme.stroke
        positions = frame["position"].values
        fractions = frame["fraction"].values

        self.ax.add_collection(LineCollection(
            [[(x, 0.0), (x, y)] for x, y in zip(positions, fractions)],
            colors=color, linewidths=self.config["stem_width"] * stroke, zorder=5,
        ))
        self.ax.scatter(
            positions, fractions, s=self.config["dot_size"] * stroke,
            facecolor=color, edgecolor="none", zorder=6, clip_on=False,
        )

    def _draw_key(self, codes: List[str]) -> None:
        """Name the codes in their own colours, so no legend box is needed."""
        theme = self.theme
        em = self._em_in_axes_fraction()
        x = 0.0
        for code in codes:
            label = mod_label(code)
            self.halo(self.ax.text(
                x, 1.0, label, transform=self.ax.transAxes,
                ha="left", va="top", color=self._color(code),
                fontsize=theme.fonts["annot"], zorder=8,
            ))
            x += (len(label) * 0.6 + self.config["key_gap_em"]) * em
        self.halo(self.ax.text(
            1.0, 1.0, "0\u2013100%", transform=self.ax.transAxes,
            ha="right", va="top", color=theme.ink["muted"],
            fontsize=theme.fonts["annot"], zorder=8,
        ))

    def _em_in_axes_fraction(self) -> float:
        """Width of one em as a fraction of the axes, for laying out the key."""
        width_in = self.ax.get_window_extent().width / self.ax.figure.dpi
        if width_in <= 0:
            return 0.02
        return (self.theme.fonts["annot"] / 72.0) / width_in

    @staticmethod
    def _site_count(shown: int, assessed: int) -> str:
        """Sites drawn against sites assessed, so the filter is visible."""
        if assessed:
            return f"{shown:,}/{assessed:,} sites"
        return f"{shown:,} site" + ("" if shown == 1 else "s")
