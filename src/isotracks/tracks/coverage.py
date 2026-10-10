"""Coverage track with sashimi junction arcs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib.patches import Polygon
from scipy import interpolate

from ..io.alignments import CoverageReader
from ..theme import INK
from .base import Track


def _bspline(x: np.ndarray, y: np.ndarray, n: int = 100, degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth curve through the control points of a junction arc."""
    control = np.asarray([x, y]).T
    count = control.shape[0]
    degree = int(np.clip(degree, 1, count - 1))
    knots = np.array(
        [0] * degree + list(range(count - degree + 1)) + [count - degree] * degree,
        dtype="int",
    )
    u = np.linspace(0, count - degree, n)
    return np.array(interpolate.splev(u, (knots, control.T, degree)))


def _runs_of_equal(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous runs of equal value as ``([[start, stop], ...], values)``."""
    edges = np.append(np.where(np.insert(np.diff(values), 0, True))[0], len(values))
    if edges[0] != 0:
        edges = np.insert(edges, 0, 0)
    if edges[-1] != len(values):
        edges = np.append(edges, len(values))
    return np.array([edges[:-1], edges[1:]]).T, values[edges[:-1]]


def _nonzero_blocks(mask: np.ndarray) -> np.ndarray:
    """Start/stop indices of each ``True`` run in a boolean array."""
    padded = np.pad(mask.astype(int), (1, 1))
    return np.where(np.diff(padded))[0].reshape(-1, 2)


class CoverageTrack(Track):
    """Read depth as a filled area, with sashimi junction arcs."""

    #: A detail panel keeps this panel's depth scale.
    PANEL_SHARED_Y = True

    DEFAULT_CONFIG: Dict[str, Any] = {
        "reader_type": CoverageReader,
        "height": 3.0,
        "show_scale": True,
        # Headroom for junction arcs, in the top margin.
        "y_margin": (0.0, 0.35),
        "color": "#1B2A49",
        "fill_alpha": 0.9,
        "junctions": True,
        "junction_color": None,  # None -> the track colour
        "min_junction_ratio": 0.08,  # hide arcs used by very few reads
        "min_junction_count": 0,
        "junction_linewidth": (0.25, 0.9),  # (min, max) in points
        # Apex height as a fraction of the headroom above the flanking depth.
        "junction_lift": (0.30, 0.95),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_ylim: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------- scaling

    def peak_depth(self) -> float:
        """Highest depth in the current region, 0 when there is no data."""
        if not self.region_data:
            return 0.0
        depth = self.region_data[0]
        if depth.empty:
            return 0.0
        return float(depth["depth"].max())

    def preferred_ylim(self) -> Tuple[float, float]:
        """Depth range this track would choose on its own."""
        return 0.0, max(1.0, self.peak_depth())

    def force_ylim(self, low: float, high: float) -> None:
        """Pin the depth range, used to put every condition on one scale."""
        self._forced_ylim = (low, high)

    # ------------------------------------------------------------ plotting

    def draw(self) -> None:
        if not self.region_data:
            self.set_ylim(0, 1)
            return
        depth, junctions = self.region_data

        low, high = self._forced_ylim or self.preferred_ylim()
        self.set_ylim(low, high)

        color = self.config["color"]
        self._fill_depth(depth, color)
        if self.config["junctions"] and junctions is not None and not junctions.empty:
            self._draw_junctions(junctions, self.ylim[1],
                                 self.config["junction_color"] or color)

    def _fill_depth(self, depth, color: str) -> None:
        if depth is None or depth.empty:
            return
        positions = depth["pos"].values
        values = depth["depth"].values
        for start, stop in _nonzero_blocks(values > 0):
            spans, levels = _runs_of_equal(values[start:stop])
            # Step outline: two x per run, padded down to the baseline at each end.
            xs = np.pad(spans.flatten() - 0.5, (1, 1), "edge") + positions[start]
            ys = np.pad(np.repeat(levels, 2), (1, 1))
            self.ax.add_patch(Polygon(
                np.array([xs, ys]).T, closed=True,
                facecolor=color, edgecolor="none",
                alpha=self.config["fill_alpha"], zorder=5,
            ))

    def _draw_junctions(self, junctions, ceiling: float, color: str) -> None:
        """Draw one arc per junction, none hidden and none off the top."""
        keep = junctions
        if self.config["min_junction_ratio"]:
            keep = keep[keep["ratio"] >= self.config["min_junction_ratio"]]
        if self.config["min_junction_count"]:
            keep = keep[keep["count"] >= self.config["min_junction_count"]]
        if keep.empty:
            return

        # Shortest intron lowest.
        keep = keep.assign(_span=(keep["end"] - keep["start"]).abs())
        keep = keep.sort_values("_span").reset_index(drop=True)

        lw_min, lw_max = self.config["junction_linewidth"]
        lift_min, lift_max = self.config["junction_lift"]
        stroke = self.theme.stroke
        count = len(keep)

        arcs = []
        for rank, row in keep.iterrows():
            left, right = float(row["start"]), float(row["end"])
            y_left, y_right = float(row["left_depth"]), float(row["right_depth"])
            ratio = float(row["ratio"])
            if not np.isfinite(ratio):
                ratio = 0.0
            width = np.clip(lw_min + (lw_max - lw_min) * ratio, lw_min, lw_max) * stroke

            share = rank / (count - 1) if count > 1 else 0.5
            lift = lift_min + (lift_max - lift_min) * share
            flank = max(y_left, y_right)
            # A fraction of the room left above.
            apex = flank + lift * max(ceiling - flank, 0.0)

            middle = (left + right) / 2.0
            left_rise = apex - y_left
            right_drop = y_right - apex
            xs = np.array([
                left, left + 0.3 * (middle - left), left + 0.8 * (middle - left), middle,
                middle + 0.2 * (right - middle), middle + 0.7 * (middle - left), right,
            ])
            ys = np.array([
                y_left, y_left + 0.7 * left_rise, y_left + 0.98 * left_rise, apex,
                apex + 0.02 * right_drop, apex + 0.3 * right_drop, y_right,
            ])
            arcs.append((width, _bspline(xs, ys)))

        for width, (curve_x, curve_y) in sorted(arcs, key=lambda a: -a[0]):
            self.ax.plot(curve_x, curve_y, color=color, linewidth=width,
                         solid_capstyle="round", zorder=4)

        # Hairline baseline.
        self.ax.axhline(0, color=INK["hairline"], linewidth=0.5 * stroke, zorder=3)
