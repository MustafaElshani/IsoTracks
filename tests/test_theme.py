"""Theme and layout: sizing, palettes, colour-vision safety, label elision."""

from __future__ import annotations

import pytest

from isotracks import theme as theme_module
from isotracks.layout import Layout, shorten_label
from isotracks.theme import MIN_PT, Theme


def test_named_widths_resolve_to_millimetres():
    assert Theme(width="single").width_mm == 89.0
    assert Theme(width="double").width_mm == 183.0
    assert Theme(width=140).width_mm == 140.0


def test_unknown_width_is_refused_with_the_valid_options():
    with pytest.raises(ValueError, match="unknown width"):
        _ = Theme(width="enormous").width_mm


def test_unknown_readability_is_refused():
    with pytest.raises(ValueError, match="unknown readability"):
        _ = Theme(readability="tiny").fonts


def test_readability_orders_type_sizes():
    strict = Theme(readability="strict").fonts
    comfortable = Theme(readability="comfortable").fonts
    large = Theme(readability="large").fonts
    assert strict["label"] <= comfortable["label"] < large["label"]


def test_no_text_falls_below_the_print_minimum():
    for level in ("strict", "comfortable", "large"):
        for size in Theme(readability=level).fonts.values():
            assert size >= MIN_PT


def test_type_hierarchy_holds():
    fonts = Theme().fonts
    assert fonts["tick"] <= fonts["label"]
    assert fonts["annot"] <= fonts["label"]


def test_isoform_colors_cycle():
    theme = Theme()
    palette = theme.isoform_colors
    assert theme.isoform_color(0) == theme.isoform_color(len(palette))


def test_two_conditions_get_the_grey_navy_pair():
    palette = Theme().condition_palette(["Ctrl", "Trt"])
    assert palette["Ctrl"] == "#9AA5B1"
    assert palette["Trt"] == "#1B2A49"


def test_three_conditions_fall_back_to_the_series():
    palette = Theme().condition_palette(["A", "B", "C"])
    assert len(set(palette.values())) == 3


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_the_condition_pair_separates_in_greyscale():
    palette = Theme().condition_palette(["Ctrl", "Trt"])
    gap = abs(_luminance(palette["Ctrl"]) - _luminance(palette["Trt"]))
    assert gap > 0.25


def _lab(hex_color: str):
    """sRGB hex to CIE L*a*b* (D65), for measuring colour difference."""
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = linear
    xyz = (
        (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047,
        (0.2126 * r + 0.7152 * g + 0.0722 * b),
        (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883,
    )
    f = [c ** (1 / 3) if c > 0.008856 else 7.787 * c + 16 / 116 for c in xyz]
    return (116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2]))


def _delta_e(first: str, second: str) -> float:
    return sum((a - b) ** 2 for a, b in zip(_lab(first), _lab(second))) ** 0.5


def test_the_polya_accent_separates_from_every_isoform_colour():
    theme = Theme()
    worst = min(_delta_e(theme.polya_color, colour) for colour in theme.isoform_colors)
    assert worst > 18, f"poly(A) accent is only dE {worst:.0f} from an isoform colour"
    assert theme.polya_color not in theme.isoform_colors


def test_the_tss_and_polya_marks_separate_from_each_other():
    theme = Theme()
    assert _delta_e(theme.tss_color, theme.polya_signal_color) > 18


def test_rcparams_keep_output_editable_and_unframed():
    rc = Theme().rc()
    assert rc["pdf.fonttype"] == 42  # TrueType, not Type 3
    assert rc["svg.fonttype"] == "none"  # live text in SVG
    assert rc["axes.spines.top"] is False
    assert rc["axes.grid"] is False


def test_use_installs_and_records_the_active_theme():
    try:
        theme_module.use(width="double", readability="large")
        assert theme_module.active().width_mm == 183.0
        assert theme_module.active().readability == "large"
    finally:
        theme_module.use()


def test_derive_leaves_the_original_alone():
    base = Theme(width="single")
    assert base.derive(width="double").width_mm == 183.0
    assert base.width_mm == 89.0


def test_figure_grows_taller_but_never_wider():
    layout = Layout()
    short = layout.figure_size(10)
    tall = layout.figure_size(60)
    assert tall[1] > short[1]
    assert tall[0] == short[0]


def test_figure_height_is_capped():
    from isotracks.layout import MAX_HEIGHT_MM

    assert Layout().figure_size(100_000)[1] == pytest.approx(MAX_HEIGHT_MM / 25.4)


def test_gutter_width_tracks_the_longest_label():
    layout = Layout()
    assert layout.gutter_width(30) > layout.gutter_width(4)
    assert 0.4 <= layout.gutter_width(4) <= 1.5


def test_short_labels_are_left_alone():
    assert shorten_label("TX_A") == "TX_A"


def test_long_labels_are_elided_in_the_middle():
    text = "ENST00000123456789.7"
    short = shorten_label("VERY_" + text, limit=12)
    assert len(short) == 12
    assert short.startswith("VERY_")
    assert short.endswith("7")
    assert "…" in short


# --------------------------------------------- the logo, on either page colour

def _load_logo_module():
    """Import docs/make_logo.py, which is an asset script, not a package."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "docs" / "make_logo.py"
    spec = importlib.util.spec_from_file_location("isotracks_make_logo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relative_luminance(color):
    from matplotlib.colors import to_rgb

    def channel(value):
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(v) for v in to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(a, b):
    high, low = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


#: White, and GitHub's dark README background.
PAGES = ("#FFFFFF", "#0D1117")


def test_the_icon_reads_on_a_light_and_a_dark_page():
    logo = _load_logo_module()
    for name, color in logo.ICON_COLORS.items():
        for page in PAGES:
            assert _contrast(color, page) >= 3.0, f"{name} {color} on {page}"


def test_each_name_ink_reads_on_its_own_page():
    logo = _load_logo_module()
    for mode, page in (("light", PAGES[0]), ("dark", PAGES[1])):
        assert _contrast(logo.COLORS[mode]["ink"], page) >= 4.5


def test_the_logo_is_drawn_without_fonts_in_the_file():
    logo = _load_logo_module()
    drawings = [logo.icon()]
    try:
        drawings += [logo.lockup("light"), logo.lockup("dark")]
    except ValueError:
        pass                                    # the name's font is not installed here
    for svg in drawings:
        assert svg.startswith("<svg") and "<text" not in svg and "font-family" not in svg
