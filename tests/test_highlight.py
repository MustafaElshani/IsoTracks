"""Region highlights: what a name resolves to, and where the band is drawn."""

from __future__ import annotations

import pytest

from isotracks import IsoTracks
from isotracks.browser import _coding_spans
from isotracks.highlight import (
    Highlight,
    normalise_target,
    parse_coordinates,
    split_spec,
)

from .conftest import CHROM, ISOFORMS


# --------------------------------------------------------------- parsing

@pytest.mark.parametrize("spelling", ["3_utr", "3'UTR", "3′ UTR", "3-utr",
                                      "utr3", "three_prime_utr", "3UTR"])
def test_every_spelling_of_the_three_prime_utr_means_the_same(spelling):
    assert normalise_target(spelling) == "3_utr"


def test_coordinates_parse_with_commas_a_chromosome_and_either_order():
    assert parse_coordinates("chr22:31,950,000-31,951,000") == ("chr22", 31_950_000, 31_951_000)
    assert parse_coordinates("200-100") == (None, 100, 200)
    assert parse_coordinates("exon_2") is None


def test_a_transcript_is_split_from_a_name_but_not_from_a_range():
    assert split_spec("exon_2:ENST0001") == ("exon_2", "ENST0001")
    assert split_spec("chr1:100-200") == ("chr1:100-200", None)
    assert split_spec("3_utr") == ("3_utr", None)


# ------------------------------------------- BED coding coordinates

def _record(strand, thick=(1099, 6400)):
    """The fixture's TX_LONG as a BED12 row: exons 1000-1300, 3000-3200, 6000-6800."""
    return {
        "chromStart": 999, "chromEnd": 6800, "strand": strand,
        "thickStart": thick[0], "thickEnd": thick[1],
        "blockSizes": "301,201,801,", "blockStarts": "0,2000,5000,",
    }


def test_the_cds_is_its_exonic_pieces_only():
    assert _coding_spans(_record("+"), "cds") == [(1100, 1300), (3000, 3200), (6000, 6400)]


def test_utrs_follow_the_strand():
    plus, minus = _record("+"), _record("-")
    assert _coding_spans(plus, "3_utr") == [(6401, 6800)]
    assert _coding_spans(plus, "5_utr") == [(1000, 1099)]
    # On the minus strand the 3' end is the low coordinate.
    assert _coding_spans(minus, "3_utr") == [(1000, 1099)]
    assert _coding_spans(minus, "5_utr") == [(6401, 6800)]


def test_a_utr_crossing_an_intron_leaves_the_intron_out():
    spans = _coding_spans(_record("+", thick=(1099, 1200)), "3_utr")
    assert spans == [(1201, 1300), (3000, 3200), (6000, 6800)]


def test_a_non_coding_record_has_no_utr():
    assert _coding_spans(_record("+", thick=(6800, 6800)), "3_utr") == []


# ------------------------------------------------ named regions

def test_exons_resolve_on_the_isoform_zoom_would_use(view_kwargs):
    view = IsoTracks(**view_kwargs)
    region = view.highlight_region("exon_2")
    assert region.transcript_id == "TX_LONG"  # the one with most exons
    assert region.spans == [ISOFORMS["TX_LONG"][1]]
    assert region.label == "exon 2"

    _, zoom_start, zoom_end, _ = view.zoom("exon_2", padding=0, tail_room=False)
    assert (zoom_start, zoom_end) == region.spans[0]


def test_first_and_last_exon(view_kwargs):
    view = IsoTracks(**view_kwargs)
    assert view.highlight_region("first_exon").spans == [ISOFORMS["TX_LONG"][0]]
    assert view.highlight_region("last_exon").spans == [ISOFORMS["TX_LONG"][-1]]


def test_an_intron_is_the_gap_between_two_exons(view_kwargs):
    region = IsoTracks(**view_kwargs).highlight_region("intron_1")
    (_, first_end), (second_start, _) = ISOFORMS["TX_LONG"][:2]
    assert region.spans == [(first_end + 1, second_start - 1)]
    assert region.label == "intron 1"


def test_a_junction_is_the_same_span_named_by_its_exons(view_kwargs):
    view = IsoTracks(**view_kwargs)
    junction = view.highlight_region("junction_1")
    assert junction.spans == view.highlight_region("intron_1").spans
    assert junction.label == "junction 1–2"


def test_the_three_prime_utr_comes_from_the_bed(view_kwargs):
    region = IsoTracks(**view_kwargs).highlight_region("3_utr")
    assert region.spans == [(6401, 6800)]
    assert region.label == "3′ UTR"


def test_a_transcript_can_be_named(view_kwargs):
    view = IsoTracks(**view_kwargs)
    assert view.highlight_region("3_utr", "TX_SHORT").spans == [(3401, 3600)]
    assert view.highlight_region("last_exon", "TX_SHORT").spans == [ISOFORMS["TX_SHORT"][-1]]


def test_a_named_region_is_labelled_in_its_isoforms_colour(view_kwargs):
    view = IsoTracks(**view_kwargs)
    region = view.highlight_region("3_utr", "TX_SHORT")
    assert region.label_color == view.isoform_color("TX_SHORT")


def test_a_range_is_taken_as_given(view_kwargs):
    view = IsoTracks(**view_kwargs)
    assert view.highlight_region((2_000, 1_500)).spans == [(1_500, 2_000)]
    assert view.highlight_region(f"{CHROM}:1,500-2,000").spans == [(1_500, 2_000)]


@pytest.mark.parametrize("target, message", [
    ("exon_9", "does not exist"),
    ("intron_3", "no intron 3"),
    ("chrZ:1-100", "is on chrZ"),
    ("promoter", "unknown highlight"),
    ("exon_two", "bad exon target"),
])
def test_bad_regions_are_refused_with_a_reason(view_kwargs, target, message):
    with pytest.raises(ValueError, match=message):
        IsoTracks(**view_kwargs).highlight_region(target)


def test_a_utr_needs_a_bed(view_kwargs):
    with pytest.raises(ValueError, match="needs a BED"):
        IsoTracks(**{**view_kwargs, "bed_file": None}).highlight_region("3_utr")


def test_a_bad_region_fails_before_anything_is_drawn(view_kwargs, monkeypatch):
    view = IsoTracks(**view_kwargs)

    def fail(*_args, **_kwargs):
        raise AssertionError("the canvas was built before the region was checked")

    monkeypatch.setattr(IsoTracks, "build", fail)
    with pytest.raises(ValueError, match="unknown highlight"):
        view.plot(highlight="promoter")


# ---------------------------------------------------------- drawing

def _bands(figure):
    from matplotlib.patches import Rectangle
    return [a for a in figure.artists if isinstance(a, Rectangle)]


def test_one_band_runs_through_every_data_track(view_kwargs):
    from isotracks.tracks import CoordinateAxisTrack, ScaleBarTrack

    view = IsoTracks(**view_kwargs)
    figure = view.plot(highlight="exon_2", highlight_style="box")
    (band,) = _bands(figure)

    tracks = view.canvas.tracks
    data = [t for t in tracks if type(t).__name__ not in
            ("ScaleBarTrack", "CoordinateAxisTrack", "HighlightLabelTrack")]
    top = max(t.ax.get_position().y1 for t in data)
    bottom = min(t.ax.get_position().y0 for t in data)
    assert band.get_y() == pytest.approx(bottom)
    assert band.get_y() + band.get_height() == pytest.approx(top)

    ruler_top = max(t.ax.get_position().y1 for t in
                    view.canvas.of_type(ScaleBarTrack) + view.canvas.of_type(CoordinateAxisTrack))
    assert band.get_y() >= ruler_top


def test_the_band_edges_land_on_the_exon_edges(view_kwargs):
    view = IsoTracks(**view_kwargs)
    figure = view.plot(highlight="exon_2", highlight_style="box")
    (band,) = _bands(figure)
    start, end = ISOFORMS["TX_LONG"][1]
    assert band.get_x() == pytest.approx(start - 0.5)
    assert band.get_width() == pytest.approx(end - start + 1)


@pytest.mark.parametrize("style, filled, outlined", [
    ("both", 1, 1), ("box", 1, 0), ("outline", 0, 1),
])
def test_styles(view_kwargs, style, filled, outlined):
    figure = IsoTracks(**view_kwargs).plot(highlight="exon_2", highlight_style=style)
    bands = _bands(figure)
    assert sum(b.get_fill() for b in bands) == filled
    assert sum(not b.get_fill() for b in bands) == outlined


def test_an_unknown_style_is_refused(view_kwargs):
    with pytest.raises(ValueError, match="highlight_style"):
        IsoTracks(**view_kwargs).plot(highlight="exon_2", highlight_style="glow")


def test_a_multi_piece_region_gets_one_band_per_piece(view_kwargs):
    figure = IsoTracks(**view_kwargs).plot(
        highlight=Highlight([(1_000, 1_300), (3_000, 3_200)], label="two"),
        highlight_style="box")
    assert len(_bands(figure)) == 2


def test_a_region_outside_the_window_draws_nothing(view_kwargs):
    figure = IsoTracks(**view_kwargs).plot(
        region=(CHROM, 5_800, 7_000, "+"), highlight="exon_1")
    assert _bands(figure) == []


def test_names_get_a_row_of_their_own_above_the_tracks(view_kwargs):
    from isotracks.tracks import HighlightLabelTrack

    view = IsoTracks(**view_kwargs)
    view.plot(highlight=["exon_2", "3_utr"])
    assert type(view.canvas.tracks[0]) is HighlightLabelTrack
    labels = [t.get_text() for t in view.canvas.tracks[0].ax.texts]
    assert labels == ["exon 2", "3′ UTR"]


def test_an_unlabelled_range_adds_no_label_row(view_kwargs):
    from isotracks.tracks import HighlightLabelTrack

    view = IsoTracks(**view_kwargs)
    view.plot(highlight=(1_500, 2_000))
    assert view.canvas.of_type(HighlightLabelTrack) == []
    assert len(_bands(view.figure)) == 2  # box and edge


def test_a_name_sits_above_its_own_band(view_kwargs):
    view = IsoTracks(**view_kwargs)
    figure = view.plot(highlight="exon_2")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    label_row = view.canvas.tracks[0]
    (text,) = label_row.ax.texts
    box = text.get_window_extent(renderer=renderer)
    band = [b for b in _bands(figure) if b.get_fill()][0].get_window_extent(renderer)

    assert box.y0 >= band.y1 - 1.0  # above it
    assert band.x0 - 1.0 <= (box.x0 + box.x1) / 2 <= band.x1 + 1.0  # over it


def test_the_figure_grows_by_what_the_label_row_needs(view_kwargs):
    plain = IsoTracks(**view_kwargs).plot()
    marked = IsoTracks(**view_kwargs).plot(highlight="exon_2")
    assert marked.get_size_inches()[1] > plain.get_size_inches()[1]


def test_floored_rows_grow_the_figure_instead_of_squeezing_the_rest(view_kwargs):
    from isotracks.layout import FRAME_MM, MM_PER_UNIT

    view = IsoTracks(**view_kwargs)
    figure = view.plot()
    expected_mm = FRAME_MM + MM_PER_UNIT * sum(view.canvas.effective_heights())
    assert figure.get_size_inches()[1] * 25.4 == pytest.approx(expected_mm, rel=1e-6)


def test_the_band_follows_a_flipped_axis(minus_view_kwargs):
    from .conftest import MINUS_ISOFORMS

    view = IsoTracks(**minus_view_kwargs)
    figure = view.plot(highlight="exon_2", highlight_style="box")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    (band,) = _bands(figure)
    box = band.get_window_extent(renderer)

    # Exon 2 in transcription order: the middle exon of TXM_LONG.
    start, end = MINUS_ISOFORMS["TXM_LONG"][1]
    axis = view.canvas.tracks[-1].ax
    # Through the flipped axis, the higher coordinate is the one on the left.
    left_px = axis.transData.transform((end + 0.5, 0))[0]
    right_px = axis.transData.transform((start - 0.5, 0))[0]
    assert left_px < right_px
    assert box.x0 == pytest.approx(left_px, abs=1.0)
    assert box.x1 == pytest.approx(right_px, abs=1.0)
