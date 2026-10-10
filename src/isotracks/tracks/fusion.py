"""Reads through a fusion: one row per read, the same row in every panel."""

from __future__ import annotations

from typing import Any, Dict

from .reads import ReadTrack


class FusionReadTrack(ReadTrack):
    """Pieces of fusion reads, each at its read's fixed row."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "rows": 0,  # reads in this condition: every panel's row count
        "show_count": False,  # the view writes the count, once, for all panels
        "max_reads": None,
    }

    def filter_rows(self) -> None:
        """Keep every piece: thinning one panel would misalign the rest."""
        self._available = len(self.region_data.df)

    def assign_rows(self) -> None:
        """Each piece at its read's row, whatever else this panel holds."""
        array = self.region_data
        frame = array.df.copy()
        row = frame["order"].astype(float).values
        frame["y_min"] = row
        frame["y_max"] = row + frame["height"].astype(float).values
        array.df = self._maybe_invert(frame)
        array.push_columns(["y_min", "y_max"])

    def _row_metrics(self, rows: int) -> Dict[str, float]:
        return super()._row_metrics(max(rows, int(self.config["rows"])))

    def draw(self) -> None:
        super().draw()
        rows = int(self.config["rows"])
        if rows > 0:
            self.set_ylim(-float(rows), 0.0)
