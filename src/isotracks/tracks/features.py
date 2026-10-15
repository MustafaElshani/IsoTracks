"""Tracks that draw :class:`~isotracks.io.features.Feature` objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex, to_rgb

from ..io._intervals import merge_intervals
from ..io.features import BedTranscripts, Feature
from .base import Track

#: Feature types rendered as a body rather than a connector.
BODY_TYPES = {"exon", "CDS", "5UTR", "3UTR", "polyA"}
#: Untranslated feature types, drawn hollow.
UTR_TYPES = {"5UTR", "3UTR"}


def _lighten(color: str, weight: float, background: str = "#FFFFFF") -> str:
    """Mix *color* toward *background*, keeping *weight* of the original."""
    weight = float(np.clip(weight, 0.0, 1.0))
    mixed = [
        c * weight + b * (1.0 - weight)
        for c, b in zip(to_rgb(color), to_rgb(background))
    ]
    return to_hex(mixed)


class _RectBatch:
    """Collects rectangles sharing a style so they draw as one artist."""

    def __init__(self):
        self._batches: Dict[Tuple[Any, ...], List[patches.Rectangle]] = {}

    def add(self, rect, facecolor, edgecolor, linewidth, zorder) -> None:
        self._batches.setdefault((facecolor, edgecolor, linewidth, zorder), []).append(rect)

    def flush(self, ax) -> None:
        for (facecolor, edgecolor, linewidth, zorder), rects in self._batches.items():
            collection = PatchCollection(
                rects, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, zorder=zorder, match_original=False,
            )
            collection.set_snap(False)
            ax.add_collection(collection)
        self._batches.clear()


class FeatureTrack(Track):
    """Stacks features into rows and draws their exon/intron structure."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "height": 1.0,
        "reader_type": None,
        "layout": "packed",  # "packed" (IGV), "ordered", or "overlay"
        "invert_y": True,  # first row at the top
        "row_gap": 0.2,  # fraction of a row left blank
        "body_height": 1.0,  # fraction of the row a solid body fills
        "thin_height": 0.45,  # fraction of the row a UTR fills
        "intron_height": 0.16,  # fraction of the row the intron hairline fills
        "min_intron_fraction": 0.008,  # widen introns below this share of the view
        "min_body_width_pt": 0.5,  # narrowest an exon is drawn, times the stroke weight
        "hollow_utr": False,  # draw untranslated blocks as outlines
        "direction_arrows": False,  # chevrons along the introns
        "arrow_spacing_in": 0.28,  # inches between direction chevrons
        # Half-height of a chevron as a share of the row.
        "arrow_reach": 0.18,
        "linewidth": 0.0,
        "outline_width": 0.5,  # stroke for hollow blocks
        "style": {
            "exon": {"facecolor": "#1A1A1A", "edgecolor": "none", "zorder": 10},
            "CDS": {"facecolor": "#1A1A1A", "edgecolor": "none", "zorder": 10},
            "5UTR": {"facecolor": "#1A1A1A", "edgecolor": "none", "zorder": 10},
            "3UTR": {"facecolor": "#1A1A1A", "edgecolor": "none", "zorder": 10},
            "polyA": {"facecolor": "#C8672A", "edgecolor": "none", "zorder": 11},
            "intron": {"facecolor": "#C0C6CE", "edgecolor": "none", "zorder": 9},
        },
    }
    _min_body = 0.0

    # ------------------------------------------------------------- filtering

    def filter_rows(self) -> None:
        """Drop rows before layout."""
        select = self.config["select_features"]
        if select and "id" in self.region_data.df:
            df = self.region_data.df
            self.region_data.df = df[df["id"].isin(select)]
            self.region_data.reorder_to_frame()

    # --------------------------------------------------------------- layout

    def assign_rows(self) -> None:
        """Give every feature a ``y_min``/``y_max`` row."""
        layout = self.config["layout"]
        if layout == "packed":
            self._pack_rows()
        elif layout == "overlay":
            self._overlay_rows()
        else:
            self._stack_rows()

    def _overlay_rows(self) -> None:
        """Every feature on one row, so the models draw on top of each other."""
        array = self.region_data
        df = array.df.copy()
        df["y_min"] = 0.0
        df["y_max"] = df["height"].values
        array.df = self._maybe_invert(df)
        array.push_columns(["y_min", "y_max"])

    def _stack_rows(self) -> None:
        """One feature per row, in the order set by :meth:`order_rows`."""
        array = self.region_data
        df = array.df
        if "order" not in df.columns:
            self.order_rows()
            df = array.df
        df = df.copy()
        order_index = np.argsort(df["order"].values, kind="stable")
        stacked = df["height"].values[order_index].cumsum()
        y_max = np.empty(len(df), dtype=float)
        y_max[order_index] = stacked
        df["y_max"] = y_max
        df["y_min"] = y_max - df["height"].values
        array.df = self._maybe_invert(df)
        array.push_columns(["y_min", "y_max"])

    def order_rows(self) -> None:
        """Default ordering: by genomic start, respecting a reversed x-axis."""
        df = self.region_data.df.sort_values(
            ["start", "end"], ascending=not self.x_inverted,
        )
        df["order"] = np.arange(len(df))
        self.region_data.df = df

    def _pack_rows(self) -> None:
        """IGV-style packing: fill each row left to right before opening a new one."""
        array = self.region_data
        df = array.df.copy()

        if self.x_inverted:
            furthest = df["end"].max()
            df["_start"] = furthest - df["end"]
            df["_end"] = furthest - df["start"]
        else:
            df["_start"] = df["start"]
            df["_end"] = df["end"]
        df = df.sort_values(["_start", "_end"])

        span = abs(self.region_end - self.region_start) or 1
        gap = max(1.0, span * 0.004)

        row_ends: List[float] = []
        rows: List[int] = []
        for start, end in zip(df["_start"].values, df["_end"].values):
            placed = False
            for row, occupied_to in enumerate(row_ends):
                if start > occupied_to + gap:
                    row_ends[row] = end
                    rows.append(row)
                    placed = True
                    break
            if not placed:
                rows.append(len(row_ends))
                row_ends.append(end)

        df["y_min"] = np.asarray(rows, dtype=float)
        df["y_max"] = df["y_min"] + df["height"].values
        df = df.drop(columns=["_start", "_end"])
        array.df = self._maybe_invert(df)
        array.push_columns(["y_min", "y_max"])

    def _maybe_invert(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config["invert_y"]:
            df["y_min"], df["y_max"] = -df["y_max"], -df["y_min"]
        return df

    # ------------------------------------------------------------- plotting

    def draw(self) -> None:
        self._dense_intron_alpha = None
        self._metrics = None
        if self.region_data is None or self.region_data.is_empty():
            self.set_ylim(0, 1)
            return

        self.filter_rows()
        if self.region_data.is_empty():
            self.set_ylim(0, 1)
            return
        self.assign_rows()

        df = self.region_data.df
        self.set_ylim(
            float(min(df["y_min"].min(), df["y_max"].min())),
            float(max(df["y_min"].max(), df["y_max"].max())),
        )

        metrics = self._metrics = self._row_metrics(len(df))
        self._min_body = self._min_body_units()
        batch = _RectBatch()
        for feature in self.region_data:
            self._draw_feature(feature, batch, metrics)
        batch.flush(self.ax)

    def intron_band(self, feature: Feature) -> Tuple[float, float, str]:
        """The ``(bottom, height, colour)`` of an intron drawn in *feature*'s row."""
        metrics = getattr(self, "_metrics", None) or self._row_metrics(1)
        row_low, row_high = self._row_span(feature, metrics)
        bottom, height = self._band(row_low, row_high, metrics["intron"])
        colour = self.config["style"]["intron"]["facecolor"]
        if self._dense_intron_alpha is not None:
            colour = _lighten(colour, self._dense_intron_alpha)
        return bottom, height, colour

    @staticmethod
    def _row_span(feature: Feature, metrics: Dict[str, float]) -> Tuple[float, float]:
        """A feature's row, less the gap kept between rows."""
        row_low, row_high = sorted((feature.y_min, feature.y_max))
        gap = metrics["gap"] * (row_high - row_low)
        return row_low + gap / 2, row_high - gap / 2

    @staticmethod
    def _band(row_low: float, row_high: float, fraction: float) -> Tuple[float, float]:
        """A part *fraction* of the row tall, centred on its midline."""
        height = (row_high - row_low) * fraction
        return row_low + (row_high - row_low - height) / 2.0, height

    def _row_metrics(self, rows: int) -> Dict[str, float]:
        """Row gap and part heights, adjusted when rows fall below a pixel."""
        metrics = {
            "gap": float(self.config["row_gap"]),
            "body": float(self.config["body_height"]),
            "thin": float(self.config["thin_height"]),
            "intron": float(self.config["intron_height"]),
        }
        if rows <= 1:
            return metrics
        height_px = self.ax.get_window_extent().height
        if height_px <= 0 or height_px / rows >= 2.0:
            return metrics
        self._dense_intron_alpha = metrics["intron"]
        return {"gap": 0.0, "body": 1.0, "thin": 1.0, "intron": 1.0}

    def _min_body_units(self) -> float:
        """The narrowest a body is drawn, in drawn units of the x-axis."""
        width_pt = float(self.config.get("min_body_width_pt") or 0.0) * self.theme.stroke
        axes_in = self.ax.get_window_extent().width / self.ax.figure.dpi
        if width_pt <= 0 or axes_in <= 0:
            return 0.0
        to_drawn, _ = self._drawn_units()
        low, high = sorted((float(self.region_start), float(self.region_end)))
        span = float(to_drawn(high) - to_drawn(low))
        return span * width_pt / 72.0 / axes_in

    def _widened(self, left: float, right: float) -> Tuple[float, float]:
        """*left* to *right*, widened about its centre to the narrowest drawn body."""
        to_drawn, to_genomic = self._drawn_units()
        low, high = float(to_drawn(left)), float(to_drawn(right))
        if high - low >= self._min_body:
            return left, right
        middle, half = (low + high) / 2.0, self._min_body / 2.0
        return float(to_genomic(middle - half)), float(to_genomic(middle + half))

    def _draw_feature(self, feature: Feature, batch: _RectBatch,
                      metrics: Optional[Dict[str, float]] = None) -> None:
        if feature.chr_name and feature.chr_name != self.chr_name:
            return
        if metrics is None:
            metrics = self._row_metrics(1)
        row_low, row_high = self._row_span(feature, metrics)

        children = self._spaced_children(feature)
        for child in children:
            self._draw_part(child, batch, row_low, row_high, metrics)
        if self.config["direction_arrows"]:
            self._draw_direction(children, feature.strand, row_low, row_high)

    def _draw_part(self, feature: Feature, batch: _RectBatch, row_low: float,
                   row_high: float, metrics: Dict[str, float]) -> None:
        style = self.config["style"].get(feature.type)
        if style is None:
            return
        linewidth = self.config["linewidth"]

        if feature.type == "intron":
            fraction = metrics["intron"]
        elif feature.type in UTR_TYPES:
            fraction = metrics["thin"]
        else:
            fraction = metrics["body"]

        bottom, part_height = self._band(row_low, row_high, fraction)
        if feature.end - feature.start + 1 <= 0:
            return
        left, right = feature.start - 0.5, feature.end + 0.5
        if feature.type in BODY_TYPES and self._min_body > 0:
            left, right = self._widened(left, right)

        facecolor = feature.parm.get("facecolor", style["facecolor"])
        edgecolor = feature.parm.get("edgecolor", style.get("edgecolor", "none"))
        if feature.type == "intron" and self._dense_intron_alpha is not None:
            facecolor = _lighten(facecolor, self._dense_intron_alpha)
        elif feature.type in UTR_TYPES and self.config["hollow_utr"]:
            # Untranslated blocks are drawn unfilled.
            edgecolor, facecolor = facecolor, "none"
            linewidth = self.config["outline_width"] * self.theme.stroke
        batch.add(
            patches.Rectangle((left, bottom), width=right - left, height=part_height),
            facecolor, edgecolor, linewidth, style.get("zorder", 10),
        )

    def _draw_direction(self, children: List[Feature], strand: Optional[str],
                        row_low: float, row_high: float) -> None:
        """Small chevrons along the introns, pointing the way the gene is read."""
        introns = [c for c in children if c.type == "intron"]
        if not introns or strand not in ("+", "-"):
            return

        to_drawn, to_genomic = self._drawn_units()
        low, high = sorted((float(self.region_start), float(self.region_end)))
        span = float(to_drawn(high) - to_drawn(low)) or 1.0
        axes_in = self.ax.get_window_extent().width / self.ax.figure.dpi
        if axes_in <= 0:
            return
        step = span * self.config["arrow_spacing_in"] / axes_in
        if step <= 0:
            return

        middle = (row_low + row_high) / 2.0
        reach = (row_high - row_low) * self.config["arrow_reach"]
        # A chevron is as wide as it is tall.
        width = span * (reach / (row_high - row_low)) * 0.016
        forward = strand == "+"

        xs, ys = [], []
        for intron in introns:
            start, end = float(to_drawn(intron.start)), float(to_drawn(intron.end))
            if end - start < step * 0.75:
                continue
            position = start + step / 2.0
            while position < end:
                tip = position + width if forward else position - width
                back = position - (tip - position)
                xs.extend([float(to_genomic(back)), float(to_genomic(tip)),
                           float(to_genomic(back)), None])
                ys.extend([middle + reach, middle, middle - reach, None])
                position += step

        if xs:
            self.ax.plot(
                xs, ys, color=self.theme.ink["rule"],
                linewidth=0.5 * self.theme.stroke, solid_capstyle="round",
                zorder=9.5,
            )

    def _spaced_children(self, feature: Feature) -> List[Feature]:
        """Copy the children, widening any intron too narrow to see."""
        children = sorted(feature.children, key=lambda f: f.start)
        min_fraction = self.config["min_intron_fraction"]
        span = abs(self.region_end - self.region_start)
        if not min_fraction or span <= 0 or self.intron_map is not None:
            return children

        min_gap = min_fraction * span
        narrow = [
            i for i, child in enumerate(children)
            if child.type == "intron" and (child.end - child.start + 1) < min_gap
        ]
        if not narrow:
            return children

        copies = [
            Feature(c.start, c.end, c.chr_name, c.strand, c.id, c.height, c.type, c.parm, c.children)
            for c in children
        ]
        spare =[max(0.0, (c.end - c.start + 1) - self._min_body) / 2.0 for c in children]
        for i in narrow:
            intron = copies[i]
            trim = (min_gap - (intron.end - intron.start + 1)) / 2.0
            left = next((j for j in range(i - 1, -1, -1) if copies[j].type in BODY_TYPES), None)
            right = next((j for j in range(i + 1, len(copies)) if copies[j].type in BODY_TYPES), None)
            if left is not None:
                cut = min(trim, spare[left])
                copies[left].end -= cut
                intron.start -= cut
            if right is not None:
                cut = min(trim, spare[right])
                copies[right].start += cut
                intron.end += cut
        return copies


class TranscriptTrack(FeatureTrack):
    """One BED12 isoform model per row."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": BedTranscripts,
        "reader_kwargs": {},
        "layout": "ordered",
        "height": 0.9,
        "row_gap": 0.30,
        "body_height": 0.80,
        # UTRs are drawn shorter than the CDS, and hollow.
        "thin_height": 0.42,
        "hollow_utr": True,
        # Direction chevrons are off for single isoform models.
        "direction_arrows": False,
        "keep_select_features": True,
    }

    def setup(self) -> None:
        theme = self.theme
        self.configure({
            "style.intron.facecolor": theme.intron_color,
            "style.polyA.facecolor": theme.polya_color,
        })


class GeneModelTrack(TranscriptTrack):
    """The isoforms the figure draws, overlaid on one row."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "layout": "overlay",
        "height": 0.9,
        "row_gap": 0.30,
        "hollow_utr": False,
        "direction_arrows": True,
        "alpha": None,  # None -> theme gene-model opacity
        "color": None,  # None -> theme gene-model colour
    }

    def setup(self) -> None:
        super().setup()
        theme = self.theme
        color = self.config.get("color") or theme.gene_model_color
        alpha = self.config.get("alpha")
        alpha = theme.gene_model_alpha if alpha is None else float(alpha)
        fill = (*to_rgb(color), alpha)
        self.configure({
            "style.exon.facecolor": fill,
            "style.CDS.facecolor": fill,
            "style.5UTR.facecolor": fill,
            "style.3UTR.facecolor": fill,
            "style.intron.facecolor": theme.intron_color,
        })

    def draw(self) -> None:
        """Draw the overlay, then one set of chevrons for the whole gene."""
        wanted = self.config["direction_arrows"]
        self.config["direction_arrows"] = False
        try:
            super().draw()
        finally:
            self.config["direction_arrows"] = wanted
        if wanted:
            self._draw_gene_direction()

    def _draw_gene_direction(self) -> None:
        if self.region_data is None or self.region_data.is_empty():
            return
        features = [f for f in self.region_data
                    if not f.chr_name or f.chr_name == self.chr_name]
        if not features:
            return

        blocks = merge_intervals(
            (child.start, child.end)
            for feature in features for child in (feature.children or [])
            if child.type in BODY_TYPES
        )
        gaps = [Feature(left_end + 1, right_start - 1, type_name="intron")
                for (_, left_end), (right_start, _) in zip(blocks, blocks[1:])
                if right_start - left_end > 1]

        first = features[0]
        row_low, row_high = sorted((first.y_min, first.y_max))
        gap = float(self.config["row_gap"]) * (row_high - row_low)
        self._draw_direction(gaps, first.strand, row_low + gap / 2,
                             row_high - gap / 2)
