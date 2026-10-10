"""A fusion drawn as an illustration: its partner genes, where they join, and
the transcript they make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.patches import Polygon, Rectangle

from ..layout import GUTTER_TEXT_X, type_units
from .base import Track

Span = Tuple[int, int]

#: An intron's drawn width, as a share of the median exon's.
INTRON_SHARE = 0.55


@dataclass
class SchematicGene:
    """One partner, as the schematic draws it."""

    name: str
    chrom: str
    strand: str
    color: str
    #: The gene as annotated, as one isoform, drawn grey.
    exons: List[Span] = field(default_factory=list)
    #: The fusion's exons in this gene, drawn in the partner's colour.
    kept: List[Span] = field(default_factory=list)
    #: Where the fusion enters and leaves the gene; None at the fusion's ends.
    acceptor: Optional[int] = None
    donor: Optional[int] = None

    def kept_in_order(self) -> List[Span]:
        """The kept exons 5' to 3'."""
        return sorted(self.kept, reverse=self.strand == "-")


def _merge(spans: Sequence[Span]) -> List[Span]:
    """Union of closed spans; spans that touch become one."""
    merged: List[List[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def gene_axis(exons: Sequence[Span], strand: str,
              intron_share: float = INTRON_SHARE) -> Callable[[float], float]:
    """Genomic position to 0-1 along the drawn gene, 5' end at 0."""
    merged = _merge(exons)
    widths = [math.sqrt(end - start + 1) for start, end in merged]
    intron = intron_share * float(np.median(widths))
    genomic, drawn, x = [], [], 0.0
    for index, ((start, end), width) in enumerate(zip(merged, widths)):
        if index:
            x += intron
        genomic += [start - 0.5, end + 0.5]
        drawn += [x, x + width]
        x += width
    knots, places = np.asarray(genomic), np.asarray(drawn) / x

    def to_x(position: float) -> float:
        share = float(np.interp(position, knots, places))
        return 1.0 - share if strand == "-" else share

    return to_x


class FusionSchematicTrack(Track):
    """Partner genes, their breakpoints and the fused transcript, across the figure's width."""

    SPANS_PANELS = True

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 3.2,
        "genes": [],  # SchematicGene, in transcript order
        "gene_gap": 0.07,  # between genes, as a share of the width
        "name_scale": 0.9,  # gene names: the size of the label row's
        "coord_scale": 0.8,  # breakpoint coordinates, of the annotation size
        "gene_pt": 5.0,  # exon height in the gene row
        "transcript_pt": 6.5,  # exon height in the transcript row
        "fan_pt": 15.0,  # from a gene's kept part to its stretch
        "gap_pt": 1.5,
        "bottom_pt": 5.0,  # clearance under the transcript, before the panels
        "row_labels": ("genes", "fusion"),  # in the gutter, beside the two rows
    }

    # ------------------------------------------------------------ sizes

    def _sizes(self) -> Dict[str, float]:
        annot = self.theme.fonts["annot"]
        return {"name": annot * float(self.config["name_scale"]),
                "coord": annot * float(self.config["coord_scale"])}

    def _rows(self) -> Dict[str, float]:
        """Each row's position in points up from the bottom of the track."""
        sizes, c = self._sizes(), self.config
        gap = float(c["gap_pt"])
        transcript = float(c["transcript_pt"])
        rows = {"transcript": float(c["bottom_pt"]) + transcript / 2.0}
        rows["fan_bottom"] = rows["transcript"] + transcript / 2.0 + gap
        rows["fan_top"] = rows["fan_bottom"] + float(c["fan_pt"])
        rows["gene"] = rows["fan_top"] + gap + float(c["gene_pt"]) / 2.0
        rows["coord"] = rows["gene"] + float(c["gene_pt"]) / 2.0 + gap * 1.5
        ranged = any(g.acceptor is not None and g.donor is not None
                     for g in self.config["genes"])
        rows["name"] = rows["coord"] + sizes["coord"] * 1.2 * (2 if ranged else 1) + gap
        rows["top"] = rows["name"] + sizes["name"] * 1.2
        return rows

    def min_height(self) -> float:
        return max(super().min_height(), type_units(self._rows()["top"]))

    # ------------------------------------------------------------ drawing

    def draw(self) -> None:
        genes: List[SchematicGene] = list(self.config["genes"])
        ax = self.ax
        height_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
        rows = self._rows()
        # Points up from the bottom, with any spare height above the names.
        ax.set_ylim(0.0, max(height_pt, rows["top"]))
        ax.set_xlim(0.0, 1.0)
        if not genes:
            return
        slots = self._slots(len(genes))
        stretches = self._transcript(genes, rows)
        tops = [self._draw_gene(gene, slot, stretch, rows)
                for gene, slot, stretch in zip(genes, slots, stretches)]
        baseline = max(tops) + float(self.config["gap_pt"])
        sizes = self._sizes()
        for gene, (left, right) in zip(genes, slots):
            ax.text((left + right) / 2.0, baseline, gene.name, ha="center", va="bottom",
                    color=gene.color, fontsize=sizes["name"], zorder=5,
                    gid="schematic-gene-name")

    def _slots(self, count: int) -> List[Tuple[float, float]]:
        gap = float(self.config["gene_gap"])
        width = (1.0 - gap * (count - 1)) / count
        return [(i * (width + gap), i * (width + gap) + width) for i in range(count)]

    def _draw_gene(self, gene: SchematicGene, slot: Tuple[float, float],
                   stretch: Tuple[float, float], rows: Dict[str, float]) -> float:
        """Draw one gene and its fan; returns the top of its label, in points."""
        ax, theme, c = self.ax, self.theme, self.config
        left, right = slot
        exons = gene.exons or gene.kept
        to_gene = gene_axis(list(exons) + list(gene.kept), gene.strand)

        def x(position: float) -> float:
            return left + (right - left) * to_gene(position)

        y, h = rows["gene"], float(c["gene_pt"])
        ax.plot([left, right], [y, y], color=theme.intron_color,
                linewidth=0.6 * theme.stroke, solid_capstyle="butt", zorder=1)
        for start, end, color, z in ([(s, e, theme.intron_color, 2) for s, e in exons]
                                     + [(s, e, gene.color, 3) for s, e in gene.kept]):
            a, b = sorted((x(start - 0.5), x(end + 0.5)))
            ax.add_patch(Rectangle((a, y - h / 2), b - a, h, facecolor=color,
                                   edgecolor="none", zorder=z, gid="schematic-exon"))

        # Where the fusion leaves or enters the gene.
        forward = gene.strand != "-"
        for point, leaves in ((gene.acceptor, False), (gene.donor, True)):
            if point is None:
                continue
            # The 3' edge of the last base kept, or the 5' edge of the first.
            edge = point + (0.5 if forward == leaves else -0.5)
            ax.plot([x(edge)] * 2, [y - h / 2 - 1.5, y + h / 2 + 1.5],
                    color=theme.ink["axis"], linewidth=0.7 * theme.stroke,
                    solid_capstyle="butt", zorder=4, gid="schematic-breakpoint")

        # The kept part fans down to its stretch of the transcript.
        kept = [x(v) for s, e in gene.kept for v in (s - 0.5, e + 0.5)]
        if kept:
            top, bottom = rows["fan_top"], rows["fan_bottom"]
            ax.add_patch(Polygon(
                [(min(kept), top), (max(kept), top), (stretch[1], bottom),
                 (stretch[0], bottom)], closed=True, facecolor=gene.color,
                alpha=0.16, edgecolor="none", zorder=1, gid="schematic-fan"))

        points = [p for p in (gene.acceptor, gene.donor) if p is not None]
        if not points:
            return rows["coord"]
        at = sum(x(p) for p in points) / len(points)
        label = f"{gene.chrom}:" + "–".join(f"{p:,}" for p in points)
        text = ax.text(at, rows["coord"], label, ha="center", va="bottom",
                       color=theme.ink["muted"], fontsize=self._sizes()["coord"],
                       zorder=5, gid="schematic-coordinate")
        # A range too wide for its gene breaks after the dash.
        if len(points) == 2 and self._extent(text).width > right - left:
            text.set_text(label.replace("–", "–\n"))
        self._keep_inside(text, left, right)
        return self._extent(text).y1

    def _extent(self, text):
        """A text's box in data units: x as the axes' 0-1, y in points."""
        renderer = self.ax.figure.canvas.get_renderer()
        return text.get_window_extent(renderer).transformed(self.ax.transData.inverted())

    def _keep_inside(self, text, left: float, right: float) -> None:
        """Slide a label along so it stays over its own gene."""
        box = self._extent(text)
        shift = 0.0
        if box.x1 > right:
            shift = right - box.x1
        if box.x0 + shift < left:
            shift = left - box.x0
        if shift:
            x, y = text.get_position()
            text.set_position((x + shift, y))

    def draw_gutter(self) -> None:
        """Name the two rows, in the gutter beside each."""
        theme, ax = self.theme, self.gutter_ax
        ax.set_ylim(self.ax.get_ylim())
        rows = self._rows()
        for label, y in zip(self.config["row_labels"], (rows["gene"], rows["transcript"])):
            ax.text(GUTTER_TEXT_X, y, label, ha="right", va="center", color=theme.ink["muted"],
                    fontsize=theme.fonts["gutter"], clip_on=False)

    def _transcript(self, genes: Sequence[SchematicGene],
                    rows: Dict[str, float]) -> List[Tuple[float, float]]:
        """Draw the fused transcript to scale; return each partner's stretch of it."""
        ax, theme, c = self.ax, self.theme, self.config
        pieces = [[(e - s + 1) for s, e in gene.kept_in_order()] for gene in genes]
        total = sum(sum(p) for p in pieces)
        if not total:
            return [(0.0, 0.0)] * len(genes)
        y, h = rows["transcript"], float(c["transcript_pt"])
        muted = dict(va="center", color=theme.ink["muted"],
                     fontsize=self._sizes()["coord"], zorder=5)
        five = ax.text(0.0, y, "5′ ", ha="left", **muted)
        three = ax.text(1.0, y, f" 3′  {total:,} nt", ha="right", **muted)
        left, right = self._extent(five).x1, self._extent(three).x0
        scale = (right - left) / total
        stretches: List[Tuple[float, float]] = []
        x = left
        for gene, lengths in zip(genes, pieces):
            start = x
            for index, length in enumerate(lengths):
                ax.add_patch(Rectangle((x, y - h / 2), length * scale, h,
                                       facecolor=gene.color, edgecolor="none", zorder=3,
                                       gid="schematic-transcript"))
                if index:  # an exon junction within the partner
                    ax.plot([x, x], [y - h / 2, y + h / 2], color="white",
                            linewidth=0.6 * theme.stroke, zorder=4)
                x += length * scale
            stretches.append((start, x))
        return stretches
