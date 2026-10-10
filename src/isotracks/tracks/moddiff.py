"""Differential modification: one stem per site, up for loss, down for gain."""

from __future__ import annotations

from typing import Any, Dict

from matplotlib.collections import LineCollection

from ..io.moddiff import ModDiffReader
from .base import Track


class ModDiffTrack(Track):
    """Change in modified fraction between two sample sets."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": ModDiffReader,
        "height": 2.6,
        "label_lines": 2,  # the site count is written during draw()
        "show_scale": False,  # the gutter states the range
        "y_margin": (0.06, 0.06),
        "loss_color": None,  # None -> theme modification-loss colour
        "gain_color": None,  # None -> theme modification-gain colour
        "stem_width": 0.8,
        "dot_size": 3.6,
        "limit": 1.0,  # the axis runs -limit .. +limit
        "half_lines": True,  # hairlines at +/- half, to read heights against
        "show_key": False,
    }

    def draw(self) -> None:
        sites = self.region_data
        limit = float(self.config["limit"])
        self.set_ylim(-limit, limit)

        theme = self.theme
        hairline = dict(color=theme.ink["hairline"],
                        linewidth=0.5 * theme.stroke, zorder=3)
        self.ax.axhline(0, **hairline)
        if self.config["half_lines"]:
            for level in (-limit / 2, limit / 2):
                self.ax.axhline(level, linestyle=(0, (2, 2)), **hairline)

        tested = getattr(sites, "tested", 0) if sites is not None else 0
        if sites is None or sites.is_empty():
            self.config["sublabel"] = self._site_count(0, tested)
            return

        self._draw_stems(sites.lost, self._loss_color())
        self._draw_stems(sites.gained, self._gain_color())
        self.config["sublabel"] = self._site_count(len(sites.frame), tested)
        if self.config["show_key"] and self.panel == 0:
            self._draw_key()

    def _loss_color(self) -> str:
        return self.config["loss_color"] or self.theme.mod_loss_color

    def _gain_color(self) -> str:
        return self.config["gain_color"] or self.theme.mod_gain_color

    def _draw_stems(self, frame, color: str) -> None:
        if frame.empty:
            return
        stroke = self.theme.stroke
        positions = frame["position"].values
        effects = frame["effect_size"].values
        self.ax.add_collection(LineCollection(
            [[(x, 0.0), (x, y)] for x, y in zip(positions, effects)],
            colors=color, linewidths=self.config["stem_width"] * stroke, zorder=5,
        ))
        self.ax.scatter(
            positions, effects, s=self.config["dot_size"] * stroke,
            facecolor=color, edgecolor="none", zorder=6, clip_on=False,
        )

    def _draw_key(self) -> None:
        """Name the two directions, each on its own side of zero."""
        theme = self.theme
        entries = [
            ("lost", self._loss_color(), 1.0, "top"),
            ("gained", self._gain_color(), 0.0, "bottom"),
        ]
        for label, color, y, valign in entries:
            self.halo(self.ax.text(
                0.0, y, label, transform=self.ax.transAxes,
                ha="left", va=valign, color=color,
                fontsize=theme.fonts["annot"], zorder=8,
            ))
        self.halo(self.ax.text(
            1.0, 1.0, f"\u00b1{int(self.config['limit'] * 100)}%",
            transform=self.ax.transAxes, ha="right", va="top",
            color=theme.ink["muted"], fontsize=theme.fonts["annot"], zorder=8,
        ))

    @staticmethod
    def _site_count(shown: int, tested: int) -> str:
        if tested:
            return f"{shown:,}/{tested:,} sites"
        return f"{shown:,} site" + ("" if shown == 1 else "s")
