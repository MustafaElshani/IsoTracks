"""Every figure reads 5' to 3', left to right, whichever strand the gene is on."""

from __future__ import annotations

import numpy as np
import pytest

from isotracks import IsoTracks
from isotracks.tracks import GeneModelTrack, ReadTrack

from .conftest import CHROM, MINUS_POLYA, MINUS_TSS

#: (fixture, strand, TSS, poly(A) site) for each orientation.
GENES = [
    pytest.param("view_kwargs", "+", 1_000, 6_800, id="plus"),
    pytest.param("minus_view_kwargs", "-", MINUS_TSS, MINUS_POLYA, id="minus"),
]


def _plotted(request, fixture, **plot_kwargs):
    view = IsoTracks(**request.getfixturevalue(fixture))
    view.plot(**plot_kwargs)
    return view


def _px(ax, x):
    """Where data coordinate *x* lands on screen, in pixels from the left."""
    return float(ax.transData.transform((x, 0))[0])


# ------------------------------------------------------------ orientation

@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_the_five_prime_end_is_on_the_left(request, fixture, strand, tss, polya):
    view = _plotted(request, fixture)
    ax = view.canvas.tracks[0].ax
    assert view.strand == strand
    assert _px(ax, tss) < _px(ax, polya)


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_the_axis_is_flipped_for_minus_and_only_for_minus(request, fixture,
                                                          strand, tss, polya):
    view = _plotted(request, fixture)
    for track in view.canvas.tracks:
        assert track.ax.xaxis_inverted() == (strand == "-")


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_exon_one_is_the_leftmost_exon(request, fixture, strand, tss, polya):
    view = _plotted(request, fixture)
    ax = view.canvas.tracks[0].ax
    for tid, exons in view.exons().groupby("transcript_id"):
        exons = exons.sort_values("exon_number")
        centres = [_px(ax, (r.start + r.end) / 2) for r in exons.itertuples()]
        assert centres == sorted(centres), f"{tid} exons are not 5' to 3' on screen"


# ------------------------------------------------------------ direction

def _chevrons(ax):
    """(back, tip) screen x of every chevron drawn on *ax*."""
    found = []
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), dtype=float)
        for index in range(0, len(xs) - 2, 4):
            back, tip = xs[index], xs[index + 1]
            if np.isfinite(back) and np.isfinite(tip):
                found.append((_px(ax, back), _px(ax, tip)))
    return found


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_chevrons_point_three_prime_on_screen(request, fixture, strand, tss, polya):
    view = _plotted(request, fixture)
    (gene_row,) = view.canvas.of_type(GeneModelTrack)
    chevrons = _chevrons(gene_row.ax)
    assert chevrons, "the gene row drew no chevrons to check"
    for back, tip in chevrons:
        assert tip > back, "a chevron points 3' to 5'"


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_the_gene_row_draws_one_set_of_chevrons_not_one_per_isoform(
        request, fixture, strand, tss, polya):
    view = _plotted(request, fixture)
    (gene_row,) = view.canvas.of_type(GeneModelTrack)
    tips = sorted(round(tip) for _, tip in _chevrons(gene_row.ax))
    assert all(b - a > 3 for a, b in zip(tips, tips[1:])), "chevrons overlap"


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_poly_a_tails_run_to_the_right(request, fixture, strand, tss, polya):
    view = _plotted(request, fixture)
    checked = 0
    for track in view.canvas.of_type(ReadTrack):
        for read in track.region_data:
            parts = read.children or []
            tails = [c for c in parts if c.type == "polyA"]
            bodies = [c for c in parts if c.type == "exon"]
            if not tails or not bodies:
                continue
            tail_left = min(_px(track.ax, c.start - 0.5) for c in tails)
            tail_left = min(tail_left, min(_px(track.ax, c.end + 0.5) for c in tails))
            body_right = max(max(_px(track.ax, c.start - 0.5), _px(track.ax, c.end + 0.5))
                             for c in bodies)
            assert tail_left >= body_right - 0.5, "a tail sits on the 5' side"
            checked += 1
    assert checked, "no tailed reads were drawn"


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_the_longest_tail_is_at_the_top_of_a_waterfall(request, fixture, strand,
                                                        tss, polya):
    view = _plotted(request, fixture, sort_style="waterfall")
    for track in view.canvas.of_type(ReadTrack):
        frame = track.region_data.df
        top = frame.loc[frame["y_max"].idxmax()]
        assert top["polya_length"] == frame["polya_length"].max()


# ------------------------------------------------ where the strand comes from

def test_a_gene_keeps_its_own_strand_whatever_its_window_says(minus_view_kwargs):
    honest = IsoTracks(**minus_view_kwargs)
    wrong = IsoTracks(**{**minus_view_kwargs,
                         "region": (CHROM, 9_500, 16_000, "+")})
    assert wrong.strand == "-"
    assert len(wrong.gene_reads.read_names) == len(honest.gene_reads.read_names) > 0


def test_plot_corrects_a_window_on_the_wrong_strand(minus_view_kwargs):
    view = IsoTracks(**minus_view_kwargs)
    view.plot(region=(CHROM, 9_500, 16_000, "+"))
    assert view.canvas.region[3] == "-"
    assert view.canvas.tracks[0].ax.xaxis_inverted()


def test_a_window_can_leave_the_strand_off(minus_view_kwargs):
    view = IsoTracks(**minus_view_kwargs)
    view.plot(region=(CHROM, 9_500, 16_000))
    assert view.canvas.region == (CHROM, 9_500, 16_000, "-")


@pytest.mark.parametrize("fixture, strand, window", [
    ("view_kwargs", "+", (CHROM, 900, 7_000)),
    ("minus_view_kwargs", "-", (CHROM, 9_500, 16_000)),
])
def test_a_window_without_a_gene_takes_its_strand_from_the_reads(
        request, fixture, strand, window):
    kwargs = {k: v for k, v in request.getfixturevalue(fixture).items()
              if k != "gene_name"}
    assert IsoTracks(region=window, **kwargs).strand == strand


def test_a_window_without_a_gene_keeps_a_strand_it_was_given(minus_view_kwargs):
    kwargs = {k: v for k, v in minus_view_kwargs.items() if k != "gene_name"}
    assert IsoTracks(region=(CHROM, 9_500, 16_000, "+"), **kwargs).strand == "+"


def test_a_window_on_another_chromosome_is_refused(minus_view_kwargs):
    with pytest.raises(ValueError, match="but the window is on chrZ"):
        IsoTracks(**{**minus_view_kwargs, "region": ("chrZ", 1, 100, "-")}).strand


# ------------------------------------------------------ named regions

def test_the_three_prime_utr_of_a_minus_gene_is_its_low_end(minus_view_kwargs):
    region = IsoTracks(**minus_view_kwargs).highlight_region("3_utr")
    assert region.spans == [(10_000, 10_599)]


def test_the_last_exon_of_a_minus_gene_is_its_low_end(minus_view_kwargs):
    _, start, end, strand = IsoTracks(**minus_view_kwargs).zoom(
        "last_exon", padding=0, tail_room=False)
    assert (start, end, strand) == (10_000, 10_800, "-")


def test_tail_room_is_added_on_the_three_prime_side(minus_view_kwargs):
    view = IsoTracks(**minus_view_kwargs)
    _, start, end, _ = view.zoom("last_exon", padding=0)
    assert start < 10_000 and end == 10_800


@pytest.mark.parametrize("fixture, strand, tss, polya", GENES)
def test_the_five_prime_curve_is_whole_at_the_edge_of_the_view(request, fixture,
                                                              strand, tss, polya):
    from isotracks.tracks import EndDensityTrack

    view = _plotted(request, fixture)
    (track,) = view.canvas.of_type(EndDensityTrack)
    density = track.region_data
    curve = np.asarray(density.five_prime, dtype=float)
    at_edge = curve[0] if strand == "+" else curve[-1]
    assert curve.max() > 0
    assert at_edge < 0.01 * curve.max(), (
        f"the 5' curve is cut at {at_edge / curve.max():.0%} of its height")
