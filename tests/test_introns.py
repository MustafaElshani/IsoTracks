"""Shortened introns: the map itself, and what it does to a rendered figure."""

from __future__ import annotations

import numpy as np
import pytest

from isotracks import IsoTracks
from isotracks.introns import build_intron_map
from isotracks.tracks import CoordinateAxisTrack, GeneModelTrack, ScaleBarTrack

from .conftest import CHROM, ISOFORMS, MINUS_POLYA, MINUS_TSS

#: The fixture's merged exons -- TX_SHORT's second exon runs past TX_LONG's -- in a window with
#: flanks either side.
EXONS = [(1_000, 1_300), (3_000, 3_600), (6_000, 6_800)]
WINDOW = (CHROM, 900, 7_100, "+")


def _widths(intron_map, edges):
    drawn = intron_map.forward(edges)
    return np.diff(drawn)


# Genomic edges of each piece: flank, exon, intron, exon, intron, exon, flank.
EDGES = [900, 999.5, 1300.5, 2999.5, 3600.5, 5999.5, 6800.5, 7100]
GENOMIC = np.diff(EDGES)


# ------------------------------------------------------------------ the map

def test_scaled_keeps_exons_and_flanks_and_shrinks_introns_by_one_factor():
    intron_map = build_intron_map(EXONS, WINDOW, "scaled")
    drawn = _widths(intron_map, EDGES)
    exons_and_flanks = [0, 1, 3, 5, 6]
    assert drawn[exons_and_flanks] == pytest.approx(GENOMIC[exons_and_flanks])
    ratios = drawn[[2, 4]] / GENOMIC[[2, 4]]
    assert ratios[0] == pytest.approx(ratios[1]) and ratios[0] < 1


def test_scaled_introns_take_the_share_asked_for():
    intron_map = build_intron_map(EXONS, WINDOW, "scaled", share=0.25)
    drawn = _widths(intron_map, EDGES)
    assert (drawn[2] + drawn[4]) / drawn.sum() == pytest.approx(0.25, abs=0.01)


def test_fixed_draws_every_intron_the_same_width():
    drawn = _widths(build_intron_map(EXONS, WINDOW, "fixed"), EDGES)
    assert drawn[2] == pytest.approx(drawn[4])
    assert drawn[2] < GENOMIC[2]


def test_an_intron_is_never_lengthened():
    exons = [(1_000, 1_300), (1_340, 5_000), (15_000, 16_000)]  # 39 bp, then 10 kb
    intron_map = build_intron_map(exons, (CHROM, 900, 16_100, "+"), "fixed")
    short = float(intron_map.forward(1_339.5) - intron_map.forward(1_300.5))
    assert short == pytest.approx(39.0)
    assert (1_301, 1_339) not in intron_map.breaks


def test_a_tiny_intron_beside_a_huge_one_stays_visible():
    exons = [(1_000, 1_300), (1_400, 2_000), (50_000, 51_000)]  # 99 bp, then 48 kb
    intron_map = build_intron_map(exons, (CHROM, 900, 51_100, "+"), "scaled")
    tiny = float(intron_map.forward(1_399.5) - intron_map.forward(1_300.5))
    total = float(intron_map.forward(51_100) - intron_map.forward(900))
    assert tiny / total > 0.005, "two exons would appear to touch"


def test_the_map_round_trips_and_is_true_scale_outside_the_window():
    intron_map = build_intron_map(EXONS, WINDOW, "scaled")
    points = np.array([-5_000, 500, 1_200, 2_000, 4_000, 6_500, 7_500, 20_000], float)
    assert intron_map.inverse(intron_map.forward(points)) == pytest.approx(points)
    past = intron_map.forward([7_100, 7_600])
    assert past[1] - past[0] == pytest.approx(500)  # tails beyond: true length


def test_nothing_to_shorten_gives_no_map():
    assert build_intron_map([(1_000, 2_000)], WINDOW, "scaled") is None
    assert build_intron_map(EXONS, WINDOW, "genomic") is None
    assert build_intron_map(EXONS, WINDOW, "scaled", share=0.9) is None


@pytest.mark.parametrize("mode, share", [("squash", 0.3), ("scaled", 0.95)])
def test_bad_settings_are_refused(mode, share):
    with pytest.raises(ValueError):
        build_intron_map(EXONS, WINDOW, mode, share)


# --------------------------------------------------------------- the figure

def _px(ax, x):
    return float(ax.transData.transform((x, 0))[0])


def test_exons_keep_their_proportions_on_screen(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(introns="scaled")
    ax = view.canvas.tracks[0].ax
    assert ax.get_xscale() == "function"
    (a0, a1), (b0, b1) = ISOFORMS["TX_LONG"][0], ISOFORMS["TX_LONG"][2]
    first = _px(ax, a1 + 0.5) - _px(ax, a0 - 0.5)
    last = _px(ax, b1 + 0.5) - _px(ax, b0 - 0.5)
    assert last / first == pytest.approx((b1 - b0 + 1) / (a1 - a0 + 1), rel=1e-3)
    intron = _px(ax, 2_999.5) - _px(ax, a1 + 0.5)
    assert intron / first < (2_999.5 - a1 - 0.5) / (a1 - a0 + 1)


def test_the_axis_says_introns_are_shortened(view_kwargs):
    for mode, says in (("genomic", False), ("scaled", True)):
        view = IsoTracks(**view_kwargs)
        view.plot(introns=mode)
        (axis,) = view.canvas.of_type(CoordinateAxisTrack)
        assert ("introns shortened" in axis._caption()) is says


def test_every_shortened_intron_gets_a_break_mark(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(introns="fixed")
    (axis,) = view.canvas.of_type(CoordinateAxisTrack)
    strokes = [line for line in axis.ax.lines if line.get_gid() == "intron-break"]
    assert len(strokes) == 2 * len(axis.intron_map.breaks) > 0


def test_the_scale_bar_is_true_for_exons(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(introns="scaled")
    (bar,) = view.canvas.of_type(ScaleBarTrack)
    ax = bar.ax
    (line,) = [l for l in ax.lines if len(l.get_xdata()) == 2
               and l.get_ydata()[0] == l.get_ydata()[1]]
    bar_px = abs(_px(ax, line.get_xdata()[1]) - _px(ax, line.get_xdata()[0]))
    stated = float(ax.texts[0].get_text().split()[0])
    stated *= 1_000 if "kb" in ax.texts[0].get_text() else 1
    a0, a1 = ISOFORMS["TX_LONG"][2]
    exon_px_per_bp = (_px(ax, a1 + 0.5) - _px(ax, a0 - 0.5)) / (a1 - a0 + 1)
    assert bar_px == pytest.approx(stated * exon_px_per_bp, rel=1e-3)


def test_chevrons_are_not_crowded_into_a_shortened_intron(view_kwargs):
    view = IsoTracks(**view_kwargs)
    figure = view.plot(introns="scaled")
    (gene_row,) = view.canvas.of_type(GeneModelTrack)
    tips = sorted(_px(gene_row.ax, x) for line in gene_row.ax.lines
                  for x in line.get_xdata()[1::4] if np.isfinite(x))
    spacing = gene_row.config["arrow_spacing_in"] * figure.dpi
    gaps = np.diff(tips)
    assert len(tips) and (not len(gaps) or gaps.min() > 0.5 * spacing)


def test_a_minus_strand_gene_still_reads_five_prime_left(minus_view_kwargs):
    view = IsoTracks(**minus_view_kwargs)
    view.plot(introns="scaled")
    ax = view.canvas.tracks[0].ax
    assert _px(ax, MINUS_TSS) < _px(ax, MINUS_POLYA)


def test_a_detail_panel_is_never_shortened(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(introns="scaled", detail="exon_2")
    track = view.canvas.tracks[0]
    assert track.ax.get_xscale() == "function"
    assert track.detail.ax.get_xscale() == "linear"


def test_a_window_with_nothing_to_shorten_is_drawn_true(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(introns="scaled", region=(CHROM, 6_100, 6_700, "+"))
    assert view.canvas.tracks[0].ax.get_xscale() == "linear"


def test_highlights_still_land_on_exon_edges(view_kwargs):
    from matplotlib.patches import Rectangle

    view = IsoTracks(**view_kwargs)
    figure = view.plot(introns="scaled", highlight="exon_3", highlight_style="box")
    figure.canvas.draw()
    (band,) = [a for a in figure.artists if isinstance(a, Rectangle)]
    box = band.get_window_extent(figure.canvas.get_renderer())
    ax = view.canvas.tracks[-1].ax
    start, end = ISOFORMS["TX_LONG"][2]
    assert box.x0 == pytest.approx(_px(ax, start - 0.5), abs=1.0)
    assert box.x1 == pytest.approx(_px(ax, end + 0.5), abs=1.0)


def test_an_unknown_mode_is_refused_before_drawing(view_kwargs):
    with pytest.raises(ValueError, match="unknown introns"):
        IsoTracks(**view_kwargs).plot(introns="squash")
