"""The visual theme: page width, type sizes, colours, stroke weights and export settings.

    from isotracks import theme

    theme.use(width="single", readability="comfortable")
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import matplotlib as mpl

MM_TO_IN = 1.0 / 25.4

#: Printed widths in millimetres.
COLUMN_WIDTHS: Dict[str, float] = {
    "single": 89.0,  # Nature single column
    "onehalf": 120.0,
    "double": 183.0,  # Nature double column
    "slide": 254.0,  # 16:9 presentation
}

#: Font-size offsets (pt) added to the 7 pt body size, plus a stroke multiplier.
READABILITY: Dict[str, Dict[str, float]] = {
    "strict": dict(label=0, tick=-1, annot=-1, gutter=-1, stroke=1.00),
    "comfortable": dict(label=1, tick=0, annot=0, gutter=0, stroke=1.15),
    "large": dict(label=3, tick=2, annot=2, gutter=2, stroke=1.40),
}

BODY_PT = 7.0  # journal body size the offsets above are relative to
MIN_PT = 5.0  # never print text smaller than this

SANS_STACK = [
    "Inter", "Helvetica Neue", "Helvetica", "Arial",
    "Liberation Sans", "DejaVu Sans",
]

#: Neutral inks for text, axis lines and rules.
INK: Dict[str, str] = {
    "text": "#1A1A1A",
    "axis": "#2B2B2B",
    "muted": "#7A808A",
    "rule": "#C9CED6",
    "hairline": "#E6E6E6",
}

#: Isoform colours: Okabe-Ito.
ISOFORM_COLORS: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]

#: Two conditions: grey and navy.
CONDITION_PAIR: List[str] = ["#9AA5B1", "#1B2A49"]
#: Three or more conditions: Okabe-Ito.
CONDITION_SERIES: List[str] = ["#9AA5B1", "#1B2A49", "#E3A33B", "#4F7CAC", "#B5543A"]

#: Poly(A) tails.
POLYA_COLOR = "#A4036F"
#: Poly(A) signal (hexamer) marks in the annotation track.
POLYA_SIGNAL_COLOR = "#4F7CAC"
#: Transcription start site marks.
TSS_COLOR = "#2E7D6B"
#: Introns and read gaps.
INTRON_COLOR = "#C0C6CE"
#: The gene model row.
GENE_MODEL_COLOR = "#5C6470"
#: Region highlights.
HIGHLIGHT_COLOR = "#5C6470"
HIGHLIGHT_ALPHA = 0.12
#: Opacity of one isoform in the gene model row.
GENE_MODEL_ALPHA = 0.24

#: Colours for RNA modification codes, in the order they are handed out.
MOD_COLORS: List[str] = [
    "#B4361A",  # rust
    "#1F6F8B",  # teal
    "#C08A00",  # dark gold
]
#: Most modification codes drawn in one track.
MAX_OVERLAID_MODS = len(MOD_COLORS)

#: Differential modification: lost in the second sample set, and gained.
MOD_LOSS_COLOR = "#7D3C98"
MOD_GAIN_COLOR = "#2E7D32"


def _clamp(value: float, floor: float = MIN_PT) -> float:
    return max(float(value), floor)


@dataclass(frozen=True)
class Theme:
    """Immutable description of an IsoTracks figure's look.

    Parameters
    ----------
    width : str or float
        Printed width: a key of :data:`COLUMN_WIDTHS` or millimetres.
    readability : str
        One of ``"strict"``, ``"comfortable"`` (default), ``"large"``.
    isoform_colors, condition_colors
        Override the built-in palettes.
    transparent : bool
        Draw on a transparent rather than white background.
    """

    width: Union[str, float] = "single"
    readability: str = "comfortable"
    isoform_colors: Sequence[str] = field(default_factory=lambda: list(ISOFORM_COLORS))
    condition_colors: Sequence[str] = field(default_factory=lambda: list(CONDITION_SERIES))
    polya_color: str = POLYA_COLOR
    polya_signal_color: str = POLYA_SIGNAL_COLOR
    tss_color: str = TSS_COLOR
    intron_color: str = INTRON_COLOR
    gene_model_color: str = GENE_MODEL_COLOR
    highlight_color: str = HIGHLIGHT_COLOR
    highlight_alpha: float = HIGHLIGHT_ALPHA
    gene_model_alpha: float = GENE_MODEL_ALPHA
    mod_colors: Sequence[str] = field(default_factory=lambda: list(MOD_COLORS))
    mod_loss_color: str = MOD_LOSS_COLOR
    mod_gain_color: str = MOD_GAIN_COLOR
    ink: Mapping[str, str] = field(default_factory=lambda: dict(INK))
    transparent: bool = False

    # ---------------------------------------------------------------- sizing

    @property
    def width_mm(self) -> float:
        if isinstance(self.width, str):
            try:
                return COLUMN_WIDTHS[self.width]
            except KeyError:
                raise ValueError(
                    f"unknown width {self.width!r}; use a millimetre value or one of "
                    f"{sorted(COLUMN_WIDTHS)}"
                ) from None
        return float(self.width)

    @property
    def width_in(self) -> float:
        return self.width_mm * MM_TO_IN

    @property
    def stroke(self) -> float:
        """Line-width multiplier for the current readability level."""
        return READABILITY[self._level]["stroke"]

    @property
    def _level(self) -> str:
        if self.readability not in READABILITY:
            raise ValueError(
                f"unknown readability {self.readability!r}; use one of {sorted(READABILITY)}"
            )
        return self.readability

    @property
    def fonts(self) -> Dict[str, float]:
        """Type sizes in points."""
        off = READABILITY[self._level]
        tick = _clamp(BODY_PT + off["tick"], MIN_PT + 1)
        return {
            "label": _clamp(BODY_PT + off["label"], tick),
            "tick": tick,
            "annot": _clamp(BODY_PT + off["annot"], MIN_PT + 1),
            "gutter": _clamp(BODY_PT + off["gutter"], MIN_PT + 1),
        }

    # --------------------------------------------------------------- colour

    def isoform_color(self, index: int) -> str:
        """Colour for the *index*-th isoform, cycling through the palette."""
        palette = list(self.isoform_colors)
        return palette[index % len(palette)]

    def isoform_palette(self, ids: Iterable[str]) -> Dict[str, str]:
        """Map transcript IDs to colours in the order given."""
        return {tid: self.isoform_color(i) for i, tid in enumerate(ids)}

    def mod_palette(self, codes: Sequence[str]) -> Dict[str, str]:
        """Map modification codes to colours, in the order given."""
        palette = list(self.mod_colors)
        return {str(code): palette[i % len(palette)] for i, code in enumerate(codes)}

    def condition_palette(self, names: Sequence[str]) -> Dict[str, str]:
        """Map condition names to colours."""
        names = list(names)
        base = CONDITION_PAIR if len(names) == 2 else list(self.condition_colors)
        return {name: base[i % len(base)] for i, name in enumerate(names)}

    # ------------------------------------------------------------ rcParams

    def rc(self) -> Dict[str, object]:
        """matplotlib rcParams implementing this theme."""
        f = self.fonts
        k = self.stroke
        bg = "none" if self.transparent else "white"
        return {
            "font.family": "sans-serif",
            "font.sans-serif": SANS_STACK,
            "font.size": f["tick"],
            "axes.labelsize": f["label"],
            "axes.titlesize": f["label"],
            "axes.titleweight": "regular",
            "axes.titlelocation": "left",
            "xtick.labelsize": f["tick"],
            "ytick.labelsize": f["tick"],
            "legend.fontsize": f["tick"],
            "text.color": self.ink["text"],
            "axes.labelcolor": self.ink["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.linewidth": 0.6 * k,
            "axes.edgecolor": self.ink["axis"],
            "axes.facecolor": bg,
            "axes.grid": False,
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.2 * k,
            "xtick.major.width": 0.6 * k,
            "xtick.major.pad": 2.0,
            "xtick.color": self.ink["axis"],
            "ytick.color": self.ink["axis"],
            "lines.linewidth": 0.8 * k,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.4 * k,
            "legend.frameon": False,
            "figure.facecolor": bg,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": bg,
            "savefig.transparent": self.transparent,
            "svg.fonttype": "none",  # live text in SVG
            "pdf.fonttype": 42,  # TrueType, not Type 3
            "ps.fonttype": 42,
            "path.simplify": True,
            "path.simplify_threshold": 0.6,
        }

    def apply(self) -> Dict[str, object]:
        """Install this theme's rcParams globally and make it active."""
        rc = self.rc()
        mpl.rcParams.update(rc)
        global _ACTIVE
        _ACTIVE = self
        return rc

    def derive(self, **changes) -> "Theme":
        """A copy of this theme with some fields replaced."""
        return replace(self, **changes)


_ACTIVE = Theme()


def active() -> Theme:
    """The theme currently in force."""
    return _ACTIVE


def use(theme: Optional[Theme] = None, **kwargs) -> Theme:
    """Activate a theme and install its rcParams.

        theme.use(width="double", readability="large")
    """
    t = theme if theme is not None else Theme(**kwargs)
    t.apply()
    return t
