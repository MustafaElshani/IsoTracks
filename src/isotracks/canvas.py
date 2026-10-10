"""The figure: a stack of tracks on a shared x-axis, in one or more panels.

    canvas = Canvas()
    canvas.add("TranscriptTrack", file_name="transcripts.bed", label="models")
    canvas.add("CoordinateAxisTrack")
    canvas.render(region=("chr1", 1000, 5000, "+"), figsize=(3.5, 2))
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ._dictutil import deep_copy_config
from .theme import active as active_theme
from .tracks import TRACK_TYPES, Track


@dataclass
class Panel:
    """One column of data axes: a genomic window."""

    region: Optional[Tuple[str, int, int, str]] = None
    #: Relative width.
    share: float = 1.0
    #: Shortened introns for this panel, or None for true scale.
    intron_map: Any = None
    #: Gene named in this panel's axis caption, when it differs from the figure's.
    gene: Optional[str] = None
    #: Whether this panel's axis carries a caption.
    caption: bool = True
    #: Caption text replacing the default.
    caption_text: Optional[str] = None
    #: Whether this panel draws a scale bar.
    scale_bar: bool = True


class Canvas:
    """A vertical stack of tracks drawn into one figure.

    Parameters
    ----------
    gutter_in, margin_in : float
        Widths in inches of the label column and the right margin.
    hspace : float
        Vertical gap between tracks as a fraction of the mean track height.
    """

    def __init__(
        self,
        gutter_in: float = 0.62,
        margin_in: float = 0.10,
        hspace: float = 0.10,
        margins: Optional[Dict[str, float]] = None,
    ):
        self.gutter_in = gutter_in
        self.margin_in = margin_in
        self.hspace = hspace
        self.margins = {"left": 0.01, "right": 0.995, "bottom": 0.02, "top": 0.98}
        if margins:
            self.margins.update(margins)

        self.tracks: List[Track] = []
        self.fig = None
        self.select_features: Optional[Sequence[str]] = None
        #: The panels, left to right.
        self.panels: List[Panel] = [Panel()]
        self.panel_gap_in = 0.14  # between neighbouring panels

    # ------------------------------------------------------ the main panel

    @property
    def region(self) -> Optional[Tuple[str, int, int, str]]:
        return self.panels[0].region

    @region.setter
    def region(self, value) -> None:
        self.panels[0].region = value

    @property
    def intron_map(self):
        return self.panels[0].intron_map

    @property
    def detail_region(self) -> Optional[Tuple[str, int, int, str]]:
        """The close-up's window, when the second panel is one."""
        return self.panels[1].region if len(self.panels) > 1 else None

    @property
    def detail_share(self) -> float:
        total = sum(panel.share for panel in self.panels)
        return self.panels[1].share / total if len(self.panels) > 1 else 0.0

    # ----------------------------------------------------------- composition

    def add(
        self,
        track: Union[str, Type[Track], Track],
        config: Optional[Dict[str, Any]] = None,
        label: str = "",
        **kwargs,
    ) -> Track:
        """Append a track."""
        if isinstance(track, Track):
            obj = track
            if config:
                obj.configure(config)
            if kwargs:
                obj.configure(kwargs)
            if label:
                obj.config["label"] = label
        else:
            cls = TRACK_TYPES[track] if isinstance(track, str) else track
            merged = dict(config or {})
            merged.update(kwargs)
            obj = cls(merged, label=label)
        obj.canvas = self
        self.tracks.append(obj)
        return obj

    def prepend(self, track: Union[str, Type[Track], Track],
                config: Optional[Dict[str, Any]] = None, label: str = "",
                **kwargs) -> Track:
        """Like :meth:`add`, but the track goes above every other."""
        obj = self.add(track, config, label, **kwargs)
        self.tracks.insert(0, self.tracks.pop())
        return obj

    def effective_heights(self) -> List[float]:
        """Each track's height as the grid will lay it out."""
        return [max(float(t.config["height"]), t.min_height(), 1e-3)
                for t in self.tracks]

    def of_type(self, track_type: Union[str, Type[Track]]) -> List[Track]:
        """Every track that is an instance of *track_type*."""
        cls = TRACK_TYPES[track_type] if isinstance(track_type, str) else track_type
        return [t for t in self.tracks if isinstance(t, cls)]

    def configure_all(self, config: Dict[str, Any],
                      track_type: Union[str, Type[Track], None] = None) -> None:
        """Merge *config* into every track, or every track of one type."""
        targets = self.tracks if track_type is None else self.of_type(track_type)
        for track in targets:
            track.configure(config)

    def close(self) -> None:
        for track in self.tracks:
            track.close()

    # ---------------------------------------------------------------- region

    def set_region(self, region: Sequence, select_features: Optional[Sequence[str]] = None) -> None:
        """Point every track at *region* = ``(chrom, start, end, strand)``."""
        chrom, start, end, strand = region
        self.region = (chrom, int(start), int(end), strand)
        self.select_features = select_features
        inverted = strand == "-"
        for track in self.tracks:
            track.set_region(self.region, select_features)
            track.xlim = (self.region[1], self.region[2])
            track.x_inverted = inverted

    def set_intron_map(self, intron_map) -> None:
        """Shorten the main panel's introns with *intron_map* (None for none)."""
        self.panels[0].intron_map = intron_map

    def set_detail(self, region: Optional[Sequence], share: float = 0.4) -> None:
        """Draw *region* again beside the main window, on the same rows."""
        del self.panels[1:]
        self.panels[0].share = 1.0
        if region is None:
            return
        if not 0.1 <= share <= 0.8:
            raise ValueError(f"detail share must be between 0.1 and 0.8, got {share}")
        chrom, start, end, strand = region
        self.panels[0].share = 1.0 - float(share)
        self.panels.append(Panel(region=(chrom, int(start), int(end), strand),
                                 share=float(share), caption=False))

    def add_panel(self, panel: Panel) -> Panel:
        """Add a panel to the right of the others."""
        chrom, start, end, strand = panel.region
        panel.region = (chrom, int(start), int(end), strand)
        self.panels.append(panel)
        return panel

    def load(self) -> None:
        """Load region data for every track without drawing."""
        for track in self.tracks:
            track.load_region_data()

    # --------------------------------------------------------------- drawing

    def _width_ratios(self, fig_width: float) -> List[float]:
        """Column widths: gutter, panels with gaps between them, margin."""
        count = len(self.panels)
        gaps = self.panel_gap_in * (count - 1)
        data_width = max(fig_width - self.gutter_in - self.margin_in - gaps, 0.5)
        total = sum(panel.share for panel in self.panels) or 1.0
        widths = [self.gutter_in]
        for index, panel in enumerate(self.panels):
            if index:
                widths.append(self.panel_gap_in)
            widths.append(data_width * panel.share / total)
        widths.append(self.margin_in)
        return widths

    def render(
        self,
        region: Optional[Sequence] = None,
        select_features: Optional[Sequence[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        fig=None,
        **fig_kwargs,
    ):
        """Draw every track and return the figure."""
        if region is not None:
            self.set_region(region, select_features)
        if self.region is None:
            raise ValueError("set a region before rendering")
        if not self.tracks:
            raise ValueError("add at least one track before rendering")

        theme = active_theme()
        if fig is None:
            if figsize is None:
                figsize = (theme.width_in, theme.width_in * 1.3)
            fig = plt.figure(figsize=figsize, **fig_kwargs)
        self.fig = fig
        fig.subplots_adjust(**self.margins, wspace=0, hspace=self.hspace)

        heights = self.effective_heights()
        widths = self._width_ratios(fig.get_figwidth())
        grid = gridspec.GridSpec(
            len(self.tracks), len(widths),
            height_ratios=heights, width_ratios=widths,
            wspace=0.0, hspace=self.hspace,
            figure=fig,
        )

        shared: List[Any] = [None] * len(self.panels)
        margin_column = len(widths) - 1

        for row, track in enumerate(self.tracks):
            gutter_ax = fig.add_subplot(grid[row, 0])
            margin_ax = fig.add_subplot(grid[row, margin_column])
            for helper in (gutter_ax, margin_ax):
                helper.set_axis_off()
                helper.set_xlim(0, 1)

            if track.SPANS_PANELS:
                self._draw_spanning(track, fig.add_subplot(grid[row, 1:margin_column]),
                                    gutter_ax, margin_ax)
                continue

            copies: List[Track] = []
            for index, panel in enumerate(self.panels):
                inverted = panel.region[3] == "-"
                ax = self._panel_axes(fig, grid[row, 1 + 2 * index], shared[index],
                                      (panel.region[1], panel.region[2]),
                                      inverted, panel.intron_map)
                shared[index] = shared[index] or ax
                if index == 0:
                    track.ax = ax
                    track.gutter_ax = gutter_ax
                    track.margin_ax = margin_ax
                    track.xlim = (panel.region[1], panel.region[2])
                    track.x_inverted = inverted
                    self._enter_panel(track, index, panel)
                    track.plot()
                else:
                    copies.append(self._draw_panel(track, ax, index, panel))
            track.panel_copies = copies
            track.detail = copies[0] if copies else None

        return fig

    def _draw_spanning(self, track: Track, ax, gutter_ax, margin_ax) -> None:
        """Draw a track that spans every panel, once, in its own 0-1 axes."""
        track.ax = ax
        track.gutter_ax = gutter_ax
        track.margin_ax = margin_ax
        track.xlim = (0.0, 1.0)
        track.x_inverted = False
        self._enter_panel(track, 0, self.panels[0])
        track.intron_map = None
        ax.set_xlim(0.0, 1.0)
        track.plot()

    def _enter_panel(self, track: Track, index: int, panel: Panel) -> None:
        track.panel = index
        track.panel_spec = panel
        track.panel_count = len(self.panels)
        track.intron_map = panel.intron_map
        track.panel_copies = []
        track.detail = None

    @staticmethod
    def _panel_axes(fig, cell, shared, xlim, inverted, intron_map=None):
        ax = (fig.add_subplot(cell, sharex=shared) if shared is not None
              else fig.add_subplot(cell))
        if shared is None:
            if intron_map is not None:
                ax.set_xscale("function", functions=intron_map.functions)
            ax.set_xlim(*xlim)
            if inverted:
                ax.invert_xaxis()
        return ax

    def _draw_panel(self, track: Track, ax, index: int, panel: Panel) -> Track:
        """Draw *track* again into panel *index*, through a copy of it."""
        other = copy.copy(track)
        other.config = deep_copy_config(track.config)
        if getattr(track, "PANEL_SHARED_Y", False):
            other.config["fix_ylim"] = tuple(track.data_ylim)
            other.config["show_scale"] = False  # same number as beside it
        if panel.gene:
            other.config["gene"] = panel.gene
        if panel.caption_text is not None and "caption" in other.config:
            other.config["caption"] = panel.caption_text
        other.set_region(panel.region, self.select_features)
        other.ax = ax
        other.xlim = (panel.region[1], panel.region[2])
        other.x_inverted = panel.region[3] == "-"
        self._enter_panel(other, index, panel)
        other.plot(gutter=False)
        return other

    def save(self, path, dpi: int = 600, **kwargs):
        """Write the figure."""
        if self.fig is None:
            raise ValueError("render the canvas before saving")
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("pad_inches", 0.02)
        self.fig.savefig(path, dpi=dpi, **kwargs)
        return path
