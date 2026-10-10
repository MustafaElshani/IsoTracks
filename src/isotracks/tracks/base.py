"""The track base class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from matplotlib import patheffects

from .._dictutil import deep_copy_config, update_dict
from ..layout import GUTTER_TEXT_X, label_block_units
from ..theme import active as active_theme


class Track:
    """Base class for everything drawn as a row of the figure."""

    #: Settings every track understands.
    BASE_CONFIG: Dict[str, Any] = {
        "height": 1.0,
        "y_margin": (0.05, 0.05),
        "fix_ylim": None,
        "show_scale": False,  # draw the "[0 - N]" depth annotation
        "label": "",
        "sublabel": "",  # a second, smaller line under the label
        # Lines the gutter label takes.
        "label_lines": 1,
        "label_color": None,  # None -> theme muted ink
        "label_weight": "regular",
        "label_anchor": 0.5,  # where in the track the label sits
        "rule_above": False,  # hairline separator above this track
        # Data source
        "reader": None,  # a live reader object
        "reader_type": None,  # class to construct when file_name is set
        "file_name": "",
        "reader_kwargs": {},
        "fixed_data": None,  # bypass the reader entirely
        "select_features": None,
        "keep_select_features": False,
    }

    #: Subclass overrides.
    DEFAULT_CONFIG: Dict[str, Any] = {}

    #: Whether a detail panel keeps this track's y-scale from the main panel.
    PANEL_SHARED_Y = False

    #: Whether this track is drawn once across every panel instead of once per panel.
    SPANS_PANELS = False

    def __init__(self, config: Optional[Dict[str, Any]] = None, label: str = ""):
        self.config: Dict[str, Any] = {}
        self.configure(deep_copy_config(self.BASE_CONFIG))
        for defaults in self._default_config_chain():
            self.configure(deep_copy_config(defaults))
        self.setup()
        if config:
            self.configure(config)
        if label:
            self.config["label"] = label

        self.canvas = None
        self.ax = None
        self.gutter_ax = None
        self.margin_ax = None
        self.region: Optional[Sequence] = None
        self.chr_name: Optional[str] = None
        self.region_start: Optional[int] = None
        self.region_end: Optional[int] = None
        self.region_strand: Optional[str] = None
        self.xlim = (0, 1)
        self.x_inverted = False
        #: Index of the panel this pass draws.
        self.panel = 0
        #: The copy of this track that drew the detail panel, if there is one.
        self.detail = None
        #: The copies that drew every panel after the first, left to right.
        self.panel_copies: List["Track"] = []
        #: The panel this pass draws, and how many the figure has.
        self.panel_spec = None
        self.panel_count = 1
        #: The intron-shortening map on this panel's axis, or None at true scale.
        self.intron_map = None
        self.region_data = None
        self.ylim = (0.0, 1.0)
        self.data_ylim = (0.0, 1.0)

        self.reader = None
        self._build_reader()

    # -------------------------------------------------------------- config

    @classmethod
    def _default_config_chain(cls):
        """Each class's ``DEFAULT_CONFIG``, base-first, so subclasses override."""
        seen = []
        for klass in reversed(cls.__mro__):
            defaults = klass.__dict__.get("DEFAULT_CONFIG")
            if defaults:
                seen.append(defaults)
        return seen

    def setup(self) -> None:
        """Hook for subclasses to adjust config before caller overrides land."""

    def _drawn_units(self):
        """Genomic-to-drawn and drawn-to-genomic functions for this panel."""
        if self.intron_map is not None:
            return self.intron_map.forward, self.intron_map.inverse
        same = lambda value: np.asarray(value, dtype=float)  # noqa: E731
        return same, same

    def min_height(self) -> float:
        """Units this track needs whatever height it was configured with."""
        return label_block_units(int(self.config.get("label_lines", 1)))

    def configure(self, config: Dict[str, Any]) -> None:
        """Merge *config* into this track's settings."""
        update_dict(self.config, config)

    @property
    def label(self) -> str:
        return self.config["label"]

    @property
    def theme(self):
        return active_theme()

    # ---------------------------------------------------------------- data

    def _build_reader(self) -> None:
        reader = self.config["reader"]
        if reader is not None:
            self.reader = reader
            return
        reader_type = self.config["reader_type"]
        if reader_type is not None and self.config["file_name"]:
            kwargs = dict(self.config["reader_kwargs"])
            kwargs["file_name"] = self.config["file_name"]
            self.reader = reader_type(**kwargs)
        else:
            self.reader = None

    def close(self) -> None:
        """Release the reader's file handles."""
        if self.reader is not None:
            try:
                self.reader.close()
            except (ValueError, OSError):
                pass

    def set_region(self, region: Sequence, select_features: Optional[Sequence[str]] = None) -> None:
        self.region = region
        self.chr_name, self.region_start, self.region_end, self.region_strand = region
        if not self.config["keep_select_features"]:
            self.config["select_features"] = select_features

    def load_region_data(self) -> None:
        """Populate ``self.region_data`` for the current region."""
        if self.config["fixed_data"] is not None:
            self.region_data = self.config["fixed_data"]
        elif self.reader is not None:
            self.reader.load_region_data(self.region, self.config["select_features"])
            self.region_data = self.reader.region_data

    # -------------------------------------------------------------- limits

    def set_ylim(self, low: Optional[float] = None, high: Optional[float] = None) -> None:
        """Set the data y-range, then the axes range with margins added."""
        if self.config["fix_ylim"] is not None:
            low, high = self.config["fix_ylim"]

        low = self.data_ylim[0] if low is None or not np.isfinite(low) else float(low)
        high = self.data_ylim[1] if high is None or not np.isfinite(high) else float(high)
        if high == low:
            high = low + 1.0

        self.data_ylim = (low, high)
        span = high - low
        bottom = low - span * self.config["y_margin"][0]
        top = high + span * self.config["y_margin"][1]
        self.ylim = (bottom, top)
        self.ax.set_ylim(bottom, top)
        self.gutter_ax.set_ylim(bottom, top)

    # ------------------------------------------------------------ plotting

    def plot(self, gutter: bool = True) -> None:
        """Load, draw and label."""
        self.load_region_data()
        self._strip_axes(self.ax)
        self.draw()
        self._draw_rule()
        if gutter:
            self.draw_gutter()
        self.draw_scale()

    def draw(self) -> None:
        """Render the data."""

    def halo(self, text, width: float = 2.4):
        """Ring a label in the page colour so data can pass under it."""
        if not self.theme.transparent:
            text.set_path_effects([
                patheffects.withStroke(linewidth=width, foreground="white"),
                patheffects.Normal(),
            ])
        return text

    @staticmethod
    def _strip_axes(ax) -> None:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)

    def _draw_rule(self) -> None:
        """Hairline separator above the track, used between isoform groups."""
        if not self.config["rule_above"]:
            return
        self.ax.axhline(
            self.ylim[1], color=self.theme.ink["hairline"],
            linewidth=0.5 * self.theme.stroke, zorder=0, clip_on=False,
        )

    def draw_gutter(self) -> None:
        """Track name in the left-hand column, right-aligned and centred."""
        label = self.label
        sublabel = self.config["sublabel"]
        if not label and not sublabel:
            return
        theme = self.theme
        ax = self.gutter_ax
        color = self.config["label_color"] or theme.ink["muted"]
        anchor = float(self.config["label_anchor"])
        text_x = GUTTER_TEXT_X

        line = theme.fonts["gutter"] * 1.25
        if sublabel and label:
            offsets = (line / 2, -line / 2)
        else:
            offsets = (0.0, 0.0)

        if label:
            ax.annotate(
                label, xy=(text_x, anchor), xycoords=ax.transAxes,
                xytext=(0, offsets[0]), textcoords="offset points",
                ha="right", va="center", color=color,
                fontsize=theme.fonts["gutter"], fontweight=self.config["label_weight"],
                annotation_clip=False,
            )
        if sublabel:
            ax.annotate(
                sublabel, xy=(text_x, anchor), xycoords=ax.transAxes,
                xytext=(0, offsets[1]), textcoords="offset points",
                ha="right", va="center", color=theme.ink["muted"],
                fontsize=theme.fonts["annot"], annotation_clip=False,
            )

    def draw_scale(self) -> None:
        """Compact depth range inside the data axes, e.g. ``0-124``."""
        if not self.config["show_scale"]:
            return
        low, high = self.data_ylim
        theme = self.theme
        text = f"{self._format_scale(low)}–{self._format_scale(high)}"
        self.halo(self.ax.text(
            0.0, 0.97, text, transform=self.ax.transAxes,
            ha="left", va="top", color=theme.ink["muted"],
            fontsize=theme.fonts["annot"], clip_on=False, zorder=8,
        ))

    @staticmethod
    def _format_scale(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:.2g}"
