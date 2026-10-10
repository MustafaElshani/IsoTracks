"""A close-up of part of the gene, drawn beside the whole of it, row for row."""

from __future__ import annotations

import pytest

from isotracks import IsoTracks
from isotracks.tracks import (
    CoordinateAxisTrack,
    CoverageTrack,
    EndDensityTrack,
    HighlightLabelTrack,
    ModificationTrack,
    ReadTrack,
)
from isotracks.tracks.axis import format_coordinate

from .conftest import CHROM, ISOFORMS, MINUS_ISOFORMS


def _detailed(view_kwargs, target="last_exon", **plot_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(detail=target, **plot_kwargs)
    return view


# ------------------------------------------------------------ the window

def test_the_last_exon_window_has_room_for_the_tails(view_kwargs):
    window, marker = IsoTracks(**view_kwargs).detail_region("last_exon", padding=100)
    start, end = ISOFORMS["TX_LONG"][-1]
    assert window[1] == start - 100
    # Longest tail is 230 nt, so the 3' side reaches past padding alone.
    assert window[2] >= end + 230
    assert marker.label == "exon 3" and marker.style == "outline"


def test_a_region_inside_the_gene_gets_no_tail_room(view_kwargs):
    window, _ = IsoTracks(**view_kwargs).detail_region("exon_2", padding=50)
    start, end = ISOFORMS["TX_LONG"][1]
    assert window[1:3] == (start - 50, end + 50)


def test_detail_takes_highlight_names_and_zoom_ends(view_kwargs):
    view = IsoTracks(**view_kwargs)
    utr, marker = view.detail_region("3_utr", padding=0)
    assert utr[1] == 6_401 and marker.label == "3′ UTR"
    end, marker = view.detail_region("3_prime")
    assert marker.label == "3′ end"
    window, _ = view.detail_region(f"{CHROM}:2,000-2,400", padding=0)
    assert window[1:3] == (2_000, 2_400)


def test_a_bad_share_is_refused(view_kwargs):
    with pytest.raises(ValueError, match="detail share"):
        IsoTracks(**view_kwargs).plot(detail="exon_2", detail_share=0.95)


# ------------------------------------------------------------ the drawing

def test_every_track_is_drawn_again_in_the_detail_panel(view_kwargs):
    view = _detailed(view_kwargs)
    window = view.canvas.detail_region
    for track in view.canvas.tracks:
        assert track.detail is not None, f"{type(track).__name__} has no detail"
        assert sorted(track.detail.ax.get_xlim()) == pytest.approx(sorted(window[1:3]))


def test_the_panels_share_rows(view_kwargs):
    view = _detailed(view_kwargs)
    for track in view.canvas.tracks:
        main, detail = track.ax.get_position(), track.detail.ax.get_position()
        assert (main.y0, main.y1) == pytest.approx((detail.y0, detail.y1))
        assert detail.x0 > main.x1, "the detail panel must sit to the right"


def test_the_track_itself_still_describes_the_whole_gene(view_kwargs):
    view = _detailed(view_kwargs)
    for track in view.canvas.tracks:
        assert tuple(track.region) == view.canvas.region
        assert track.detail is not track and track.detail.config is not track.config


def test_measured_scales_are_shared_across_panels(view_kwargs):
    view = _detailed(view_kwargs)
    for track in view.canvas.of_type(CoverageTrack) + view.canvas.of_type(EndDensityTrack):
        assert tuple(track.detail.data_ylim) == pytest.approx(tuple(track.data_ylim))


def test_the_gutter_is_drawn_once(view_kwargs):
    plain = IsoTracks(**view_kwargs)
    plain.plot()
    detailed = _detailed(view_kwargs)
    for one, two in zip(plain.canvas.of_type(ReadTrack), detailed.canvas.of_type(ReadTrack)):
        assert len(two.gutter_ax.texts) == len(one.gutter_ax.texts)


def test_the_caption_and_keys_are_drawn_once(view_kwargs, modkit_pileups):
    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    view.plot(detail="last_exon")
    (axis,) = view.canvas.of_type(CoordinateAxisTrack)
    assert any("strand" in t.get_text() for t in axis.ax.texts)
    assert not any("strand" in t.get_text() for t in axis.detail.ax.texts)
    keyed = [t for t in view.canvas.of_type(ModificationTrack) if t.config["show_key"]]
    assert keyed and all(not t.detail.ax.texts for t in keyed)


def test_the_region_is_named_over_both_panels(view_kwargs):
    view = _detailed(view_kwargs)
    (labels,) = view.canvas.of_type(HighlightLabelTrack)
    assert [t.get_text() for t in labels.ax.texts] == ["exon 3"]
    assert [t.get_text() for t in labels.detail.ax.texts] == ["exon 3"]


def test_the_detail_is_framed_and_marked_by_a_dotted_outline(view_kwargs):
    from matplotlib.patches import Rectangle

    view = _detailed(view_kwargs)
    outlines = [a for a in view.figure.artists
                if isinstance(a, Rectangle) and not a.get_fill()]
    assert len(outlines) == 2  # the marker, and the frame
    assert not [a for a in view.figure.artists
                if isinstance(a, Rectangle) and a.get_fill()], "the marker is not filled"


def test_a_minus_strand_detail_still_reads_five_prime_left(minus_view_kwargs):
    view = _detailed(minus_view_kwargs)
    assert view.canvas.tracks[0].detail.ax.xaxis_inverted()
    start, end = MINUS_ISOFORMS["TXM_LONG"][0]  # the 3' exon is the lowest
    window = view.canvas.detail_region
    assert window[1] < start and window[2] == end + 200


def test_a_figure_without_detail_is_one_panel(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot()
    assert view.canvas.detail_region is None
    assert all(track.detail is None for track in view.canvas.tracks)


# ------------------------------------------------------------ axis labels

def _label_boxes(ax):
    renderer = ax.figure.canvas.get_renderer()
    return sorted((t.get_window_extent(renderer) for t in ax.texts
                   if "strand" not in t.get_text()), key=lambda b: b.x0)


@pytest.mark.parametrize("detail", [None, "last_exon"])
def test_axis_labels_never_touch(view_kwargs, detail):
    view = IsoTracks(**view_kwargs)
    figure = view.plot(detail=detail, width=40)  # narrow on purpose
    figure.canvas.draw()
    (axis,) = view.canvas.of_type(CoordinateAxisTrack)
    panels = [axis.ax] + ([axis.detail.ax] if detail else [])
    everything = []
    for ax in panels:
        boxes = _label_boxes(ax)
        assert boxes, "a panel lost every label"
        for left, right in zip(boxes, boxes[1:]):
            assert right.x0 > left.x1, "two labels overlap"
        everything += boxes
    everything.sort(key=lambda b: b.x0)
    for left, right in zip(everything, everything[1:]):
        assert right.x0 > left.x1, "labels meet across the gap between panels"


def test_whole_kilobase_ticks_drop_the_empty_decimal():
    ticks = [54_330_000.0, 54_340_000.0]
    assert format_coordinate(54_330_000, 26_000, ticks) == "54,330 kb"
    assert format_coordinate(54_335_500, 26_000, [54_335_500.0]) == "54,335.5 kb"
    assert format_coordinate(1_500, 900, [1_500.0]) == "1,500"
