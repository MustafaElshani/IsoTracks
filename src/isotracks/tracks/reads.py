"""The long-read stack: one row per read, ordered by poly(A) tail length."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..io.longreads import LongReadBam
from .features import FeatureTrack


class ReadTrack(FeatureTrack):
    """Individual long-read alignments with their poly(A) tails."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": LongReadBam,
        "layout": "ordered",
        "sort_style": "waterfall",
        "height": 3.0,
        "row_gap": 0.10,
        "body_height": 0.92,
        "thin_height": 0.92,  # reads have no UTR distinction
        "intron_height": 0.22,
        "chevron": False,  # far too small to read at one row per read
        "min_intron_fraction": 0.0,  # gaps are drawn at true width
        "show_untailed": True,  # keep reads with no poly(A) estimate
        "max_reads": None,  # cap the stack; None keeps every read
        "label_lines": 2,  # the read count is written during draw()
        "show_count": True,  # state how many reads the stack holds
    }

    def setup(self) -> None:
        theme = self.theme
        self.configure({
            "style.intron.facecolor": theme.intron_color,
            "style.polyA.facecolor": theme.polya_color,
        })

    @property
    def read_count(self) -> int:
        """Reads drawn in this stack."""
        if self.region_data is None:
            return 0
        return len(self.region_data.df)

    def draw(self) -> None:
        self._available = 0
        super().draw()
        if self.config["show_count"]:
            self.config["sublabel"] = self._count_text()

    def _count_text(self) -> str:
        """How many reads this stack holds, stated rather than implied."""
        drawn = self.read_count
        if not drawn:
            return ""
        if self._available and self._available > drawn:
            return f"n = {drawn:,}/{self._available:,}"
        return f"n = {drawn:,}"

    def filter_rows(self) -> None:
        df = self.region_data.df
        if not self.config["show_untailed"] and "read_type" in df.columns:
            df = df[df["read_type"] == "polya"]
        self._available = len(df)

        limit = self.config["max_reads"]
        if limit is not None and len(df) > limit:
            if "polya_length" in df.columns:
                df = df.sort_values("polya_length", ascending=False)
            keep = np.linspace(0, len(df) - 1, int(limit)).round().astype(int)
            df = df.iloc[np.unique(keep)]

        self.region_data.df = df
        self.region_data.reorder_to_frame()

    def order_rows(self) -> None:
        """Rank reads by poly(A) length, then lay them out in the chosen shape."""
        df = self.region_data.df
        if df.empty:
            return

        strand = self.region_strand or "+"
        tie_break = "end" if strand == "+" else "start"
        keys, ascending = ["polya_length"], [False]
        if tie_break in df.columns:
            keys.append(tie_break)
            ascending.append(strand != "+")
        if "polya_length" not in df.columns:
            keys, ascending = [tie_break], [strand != "+"]

        ranked = df.sort_values(keys, ascending=ascending, kind="stable")
        count = len(ranked)

        if self.config["sort_style"] == "diamond" and count > 2:
            slots = _diamond_slots(count)
        else:
            slots = np.arange(count)

        order = pd.Series(slots, index=ranked.index, name="order")
        df = df.copy()
        df["order"] = order
        self.region_data.df = df.sort_values("order", kind="stable")


def _diamond_slots(count: int) -> np.ndarray:
    """Row slots spiralling outward from the centre."""
    centre = count // 2
    slots: List[int] = [centre]
    offset = 1
    while len(slots) < count:
        below = centre + offset
        if below < count:
            slots.append(below)
        above = centre - offset
        if above >= 0 and len(slots) < count:
            slots.append(above)
        offset += 1
    return np.asarray(slots[:count])
