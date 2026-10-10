"""End-site annotation track, for poly(A) clusters and TSS clusters alike."""

from __future__ import annotations

from typing import Any, Dict

from matplotlib.collections import LineCollection

from ..io.clusters import ClusterSiteReader
from .base import Track

#: Height of one character of rotated text, in ems.
LABEL_LINE_EM = 0.62


class SiteTrack(Track):
    """Clustered end sites with their sequence motifs."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": ClusterSiteReader,
        "height": 0.85,
        "tick_color": None,  # None -> theme poly(A) signal colour
        "tick_width": 0.9,
        "tick_bottom": 0.10,  # as a fraction of the track
        "tick_top": 0.55,
        "show_motif": True,
        "label_gap": 0.06,  # clearance between the tick and its motif
        # Motif labels are translucent, so overlapping ones stay legible.
        "label_alpha": 0.72,
    }

    def draw(self) -> None:
        self.set_ylim(0, 1)
        sites = self.region_data
        if sites is None or len(sites) == 0:
            return

        theme = self.theme
        tick_color = self.config["tick_color"] or theme.polya_signal_color
        text_color = self.config["text_color"] if "text_color" in self.config \
            else theme.ink["muted"]
        bottom = self.config["tick_bottom"]
        top = self.config["tick_top"]

        self.ax.add_collection(LineCollection(
            [[(float(site["site"]), bottom), (float(site["site"]), top)]
             for _, site in sites.iterrows()],
            colors=tick_color,
            linewidths=self.config["tick_width"] * theme.stroke,
            zorder=6,
        ))

        if not self.config["show_motif"]:
            return
        for _, site in sites.iterrows():
            motif = site.get("motif")
            if not motif:
                continue
            self.ax.text(
                float(site["site"]), top + self.config["label_gap"], str(motif),
                color=text_color, fontsize=theme.fonts["annot"],
                alpha=self.config["label_alpha"],
                ha="center", va="bottom", rotation=90, zorder=7,
            )

    @staticmethod
    def label_units(characters: float, font_pt: float, unit_mm: float) -> float:
        """Layout units a vertical label of *characters* occupies."""
        return characters * LABEL_LINE_EM * font_pt * 25.4 / 72.0 / unit_mm
