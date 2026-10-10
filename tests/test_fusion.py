"""Fusion transcripts: parsing IsoForge's row, finding reads, drawing panels."""

from __future__ import annotations

import os

import numpy as np
import pytest

from isotracks import theme
from isotracks.annotation import GtfIndex
from isotracks.fusion import (
    FusionRead,
    collect_fusion_reads,
    complete_reads,
    fusion_model,
    junction_support,
    observed_ends,
    parse_breakpoints,
    parse_fusion_line,
    read_fusion_table,
    resolve_partners,
)
from isotracks.fusion_view import FusionView

from .conftest import FUSION_NAME, FUSION_ROW

#: A breakpoint line.
ISOFORGE_ROW = ("RAD23B::RIMS2::ATP6V1C1                     chr9::chr8::chr8            "
                "chr9:107283561>chr8:103500557;chr8:103501062>chr8:103062955"
                "                                neither;both     3               759"
                "               PASS")


# ------------------------------------------------------------------ parsing

def test_the_isoforge_row_parses():
    call = parse_fusion_line(ISOFORGE_ROW)
    assert call.genes == ["RAD23B", "RIMS2", "ATP6V1C1"]
    assert call.chromosomes == ["chr9", "chr8", "chr8"]
    first, second = call.junctions
    assert (first.donor_chrom, first.donor, first.acceptor_chrom, first.acceptor) == (
        "chr9", 107_283_561, "chr8", 103_500_557)
    assert (second.donor, second.acceptor) == (103_501_062, 103_062_955)
    assert [j.splice_sites for j in call.junctions] == ["neither", "both"]
    assert call.reads == 3 and call.status == "PASS"
    assert call.fields["unused"] == "759"


@pytest.mark.parametrize("row, message", [
    ("A::B chr1::chr2 chr1:10>chr2:20;chr2:30>chr2:40", "need 1 junctions"),
    ("A::B chr1::chr2 chr1:10>chr3:20", "partners are on chr1 and chr2"),
    ("A::B chr1::chr2 chr1:10-chr2:20", "bad breakpoint"),
    ("A::B", "at least name"),
])
def test_bad_rows_are_refused_with_a_reason(row, message):
    with pytest.raises(ValueError, match=message):
        parse_fusion_line(row)


def test_breakpoints_accept_commas():
    (junction,) = parse_breakpoints("chr1:1,000>chr2:2,000")
    assert (junction.donor, junction.acceptor) == (1_000, 2_000)


def test_a_fusion_is_found_by_name_in_its_table(fusion_kwargs):
    call = read_fusion_table(fusion_kwargs["fusion_table"], FUSION_NAME)
    assert call.genes == ["GENE5", "GENEM", "GENE3"]
    with pytest.raises(ValueError, match="is not in"):
        read_fusion_table(fusion_kwargs["fusion_table"], "NOPE::NADA")


# ----------------------------------------------------------------- partners

def _partners(fusion_kwargs):
    call = parse_fusion_line(FUSION_ROW)
    return call, resolve_partners(call, GtfIndex(fusion_kwargs["reference_gtf"]))


def test_each_partner_keeps_the_part_between_its_breakpoints(fusion_kwargs):
    _, (five, middle, three) = _partners(fusion_kwargs)
    assert (five.five_prime, five.three_prime) == (1_000, 3_200)
    # Minus strand: kept from its 5' end at 8,400 down to 8,100.
    assert (middle.strand, middle.five_prime, middle.three_prime) == ("-", 8_400, 8_100)
    assert (three.five_prime, three.three_prime) == (4_000, 4_600)


def test_breakpoints_that_run_against_the_gene_are_refused(fusion_kwargs):
    call = parse_fusion_line(FUSION_ROW.replace("chrB:8400;chrB:8100", "chrB:8100;chrB:8400"))
    with pytest.raises(ValueError, match="runs 3' to 5'"):
        resolve_partners(call, GtfIndex(fusion_kwargs["reference_gtf"]))


# -------------------------------------------------------------------- reads

def _reads(fusion_kwargs):
    call, partners = _partners(fusion_kwargs)
    return call, partners, collect_fusion_reads(partners, call.junctions,
                                                fusion_kwargs["bam_files"])


def test_reads_are_found_through_their_junctions(fusion_kwargs):
    call, _, found = _reads(fusion_kwargs)
    everything = [r for reads in found.values() for r in reads]
    assert junction_support(everything, 2) == [4, 11]
    assert {c: len(complete_reads(r, 2)) for c, r in found.items()} == {"Ctrl": 0, "Trt": 4}


def test_plain_partner_reads_do_not_count(fusion_kwargs):
    _, _, found = _reads(fusion_kwargs)
    names = {r.name for reads in found.values() for r in reads}
    assert not any("plain" in name for name in names)


def test_the_outer_ends_follow_most_reads_not_the_furthest():
    from isotracks.fusion import Partner

    first = Partner(0, "A", "chr1", "+", 1_000, 2_000)
    last = Partner(1, "B", "chr2", "+", 5_000, 9_000)
    reads = [FusionRead(f"r{i}", "c", "s", {0: [(1_100, 2_000)], 1: [(5_000, 5_500)]},
                        {0}) for i in range(19)]
    reads.append(FusionRead("runaway", "c", "s", {0: [(1_100, 2_000)],
                                                  1: [(5_000, 60_000)]}, {0}))
    observed_ends([first, last], reads)
    assert first.five_prime == 1_100
    assert last.three_prime == 5_500, "one runaway read stretched the panel"


def test_the_fusion_model_is_what_most_reads_agree_on(fusion_kwargs):
    call, partners, found = _reads(fusion_kwargs)
    model = fusion_model(partners, complete_reads(found["Trt"], 2))
    exons = {f.id: [(c.start, c.end) for c in f.children if c.type == "exon"]
             for f in model}
    assert exons["GENE5"] == [(1_000, 1_300), (3_000, 3_200)]
    assert exons["GENEM"] == [(8_100, 8_400)]
    assert exons["GENE3"] == [(4_000, 4_600)]


# ---------------------------------------------------------------- groups

def test_reads_are_grouped_one_pile_per_transcript(fusion_kwargs):
    from isotracks.fusion import group_reads

    call, partners, found = _reads(fusion_kwargs)
    groups = group_reads(partners, call.junctions, [r for rs in found.values() for r in rs])
    assert [(g.name, g.complete, len(g.reads)) for g in groups] == [
        ("GENE5::GENEM::GENE3", True, 4),  # the whole fusion comes first
        ("GENEM::GENE3", False, 7),
    ]
    assert groups[0].junctions == [0, 1] and groups[1].junctions == [1]


def test_a_partial_read_reaching_past_the_join_is_counted_as_distinct(fusion_kwargs):
    from isotracks.fusion import group_reads

    call, partners, found = _reads(fusion_kwargs)
    partial = group_reads(partners, call.junctions,
                          [r for rs in found.values() for r in rs])[1]
    assert partial.distinct == 2


# ------------------------------------------------------------------- figure

@pytest.fixture
def drawn(fusion_kwargs):
    view = FusionView(**fusion_kwargs)
    view.plot()
    return view


def _px(ax, x):
    return float(ax.transData.transform((x, 0))[0])


def _stacks(view, group_index):
    """The read tracks of one group, in condition order."""
    from isotracks.tracks import FusionReadTrack, SpacerTrack

    tracks, group = [], -1
    for track in view.canvas.tracks:
        if isinstance(track, SpacerTrack) and track.config.get("heading"):
            group += 1
        elif isinstance(track, FusionReadTrack) and group == group_index:
            tracks.append(track)
    return tracks


def _panel_row(view):
    """The first row drawn once per panel (the schematic spans them all)."""
    return next(t for t in view.canvas.tracks if not t.SPANS_PANELS)


def test_one_panel_per_partner_in_transcript_order(drawn):
    panels = drawn.canvas.panels
    assert [p.region[0] for p in panels] == ["chrA", "chrB", "chrB"]
    row = _panel_row(drawn)
    axes = [row.ax] + [c.ax for c in row.panel_copies]
    assert len(axes) == 3
    lefts = [ax.get_position().x0 for ax in axes]
    assert lefts == sorted(lefts)


def test_each_panel_reads_five_prime_to_three_prime(drawn):
    track = _panel_row(drawn)
    axes = [track.ax] + [c.ax for c in track.panel_copies]
    assert [ax.xaxis_inverted() for ax in axes] == [False, True, False]
    middle = axes[1]
    assert _px(middle, 8_400) < _px(middle, 8_100), "GENEM's 5' end must be on the left"


def test_each_transcript_gets_its_own_heading_and_piles(drawn):
    from isotracks.tracks import SpacerTrack

    headings = [t.config["heading"] for t in drawn.canvas.of_type(SpacerTrack)
                if t.config.get("heading")]
    assert headings[0] == "GENE5::GENEM::GENE3"
    assert headings[1] == "GENEM::GENE3 \u00b7 2 of 7 reads run past the GENE5 join"
    assert [t.config["rows"] for t in _stacks(drawn, 0)] == [0, 4]  # Ctrl, Trt
    assert [t.config["rows"] for t in _stacks(drawn, 1)] == [5, 2]


def test_complete_on_request_draws_only_the_whole_fusion(fusion_kwargs):
    view = FusionView(**{**fusion_kwargs, "reads": "complete"})
    view.plot()
    assert [g.name for g in view.groups] == ["GENE5::GENEM::GENE3"]


def test_a_heading_is_drawn_once_across_the_panels(drawn):
    from isotracks.tracks import SpacerTrack

    for track in drawn.canvas.of_type(SpacerTrack):
        if track.config.get("heading"):
            assert len(track.ax.texts) == 1
            assert all(not copy.ax.texts for copy in track.panel_copies)


def test_reads_that_reach_past_the_join_lead_their_pile_in_every_panel(drawn):
    ctrl = _stacks(drawn, 1)[0]  # GENEM::GENE3, Ctrl: 3 + 2
    tracks = [ctrl] + ctrl.panel_copies
    for track in tracks[1:]:  # GENEM and GENE3 panels
        frame = track.region_data.df
        top_two = frame.sort_values("y_max", ascending=False)["read_name"].head(2)
        assert all("reaching" in name for name in top_two)

    assert tracks[0].region_data.df.empty, "a partial GENEM::GENE3 read has no GENE5 piece"
    rows = {}
    for track in tracks[1:]:
        frame = track.region_data.df
        for name, y in zip(frame["read_name"], frame["y_min"]):
            rows.setdefault(name, set()).add(float(y))
    assert all(len(ys) == 1 for ys in rows.values()), "a read changed row between panels"


def test_every_breakpoint_is_marked_in_both_its_panels(drawn):
    lines = [a for a in drawn.figure.artists if a.get_gid() == "fusion-breakpoint"]
    assert len(lines) == 2 * 2 * 3
    edges = sorted({line.get_xdata()[0] for line in lines})
    assert edges == pytest.approx([3_200, 4_000, 8_100, 8_400], abs=0.5)


def _drawn_headings(view):
    from isotracks.tracks import SpacerTrack

    return [t.ax.texts[0] for t in view.canvas.of_type(SpacerTrack)
            if t.config.get("heading")]


def test_breakpoint_lines_do_not_strike_through_the_headings(drawn):
    figure = drawn.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lines = [a for a in figure.artists if a.get_gid() == "fusion-breakpoint"]
    for text in _drawn_headings(drawn):
        box = text.get_window_extent(renderer)
        for line in lines:
            (x, low), (_, high) = line.get_transform().transform(line.get_xydata())
            crosses = box.x0 <= x <= box.x1 and low < box.y1 and high > box.y0
            assert not crosses, f"a breakpoint line runs through {text.get_text()!r}"


def test_a_heading_too_long_for_the_figure_falls_back_to_its_short_form(drawn):
    from isotracks.canvas import Canvas

    assert _drawn_headings(drawn)[1].get_text() == (
        "GENEM::GENE3 \u00b7 2 of 7 reads run past the GENE5 join")
    assert drawn._heading(drawn.groups[1])[1] == "GENEM::GENE3 \u00b7 2 of 7 run past the join"

    canvas = Canvas()
    long = canvas.add("SpacerTrack", heading="a heading far too long for the figure " * 6,
                      heading_short="short")
    fits = canvas.add("SpacerTrack", heading="fits", heading_short="short")
    canvas.set_region(("chr1", 1, 1_000, "+"))
    figure = canvas.render(figsize=(3.5, 1.0))
    assert long.ax.texts[0].get_text() == "short"
    assert fits.ax.texts[0].get_text() == "fits"
    assert long.ax.texts[0].get_window_extent().x1 <= figure.bbox.x1


def test_the_fusion_gene_row_arcs_each_junction_with_every_read_across_it(drawn):
    arcs = [a for a in drawn.figure.artists if a.get_gid() == "fusion-junction"]
    counts = [t.get_text() for t in drawn.figure.texts
              if t.get_gid() == "fusion-junction-count"]
    assert len(arcs) == 2 and counts == ["4", "11"]


def test_junction_counts_clear_every_text_above_them(drawn):
    from isotracks.tracks import HighlightLabelTrack, SpacerTrack

    drawn.figure.canvas.draw()
    renderer = drawn.figure.canvas.get_renderer()
    others = [t.get_window_extent(renderer)
              for track in drawn.canvas.of_type(HighlightLabelTrack)
              + drawn.canvas.of_type(SpacerTrack)
              for panel in [track] + track.panel_copies for t in panel.ax.texts]
    for text in drawn.figure.texts:
        if text.get_gid() == "fusion-junction-count":
            count = text.get_window_extent(renderer)
            assert all(not count.overlaps(box) for box in others), text.get_text()


def test_the_tail_is_drawn_once_at_the_three_prime_end(drawn):
    trt = _stacks(drawn, 0)[1]
    tracks = [trt] + trt.panel_copies
    tailed = [sum(any(c.type == "polyA" for c in f.children) for f in t.region_data)
              for t in tracks]
    assert tailed == [0, 0, 4]


def test_partners_are_drawn_in_their_own_colours(drawn):
    assert len({p.color for p in drawn.partners}) == 3
    from isotracks.tracks import TranscriptTrack

    whole = next(t for t in drawn.canvas.of_type(TranscriptTrack)
                 if type(t) is TranscriptTrack)  # the whole fusion's row
    colours = [c.parm.get("facecolor") for f in whole.region_data for c in f.children
               if c.type == "exon"]
    assert set(colours) == {drawn.partners[0].color}  # this panel is GENE5


def test_a_whole_row_can_be_passed_instead_of_a_name(fusion_kwargs):
    kwargs = {k: v for k, v in fusion_kwargs.items() if k != "fusion_table"}
    view = FusionView(**{**kwargs, "fusion": FUSION_ROW})
    assert view.call.genes == ["GENE5", "GENEM", "GENE3"]


def test_a_name_without_a_table_is_refused(fusion_kwargs):
    kwargs = {k: v for k, v in fusion_kwargs.items() if k != "fusion_table"}
    with pytest.raises(ValueError, match="pass the fusion_table"):
        FusionView(**kwargs)


# -------------------------------------------------------- IsoForge's tables

def test_the_tables_are_found_from_a_prefix_a_directory_or_one_file(fusion_tables):
    import os
    from isotracks.fusion import FusionTables

    for source in (fusion_tables, os.path.dirname(fusion_tables),
                   fusion_tables + "_isoforge_fusion_reads.parquet"):
        assert FusionTables(source).paths["calls"].endswith("_isoforge_fusions.parquet")


def test_a_fusion_is_found_by_name_or_id(fusion_tables):
    from isotracks.fusion import FusionTables

    tables = FusionTables(fusion_tables)
    assert tables.find("GENE5::GENEM::GENE3") == ("M000001", [])
    assert tables.find("F000002") == ("F000002", [])
    with pytest.raises(ValueError, match="no fusion"):
        tables.find("NOPE::NADA")


def test_partners_come_from_the_parts_table_with_their_gene_spans(fusion_tables):
    from isotracks.fusion import FusionTables

    tables = FusionTables(fusion_tables)
    call = tables.call("M000001")
    assert [(j.donor, j.acceptor, j.splice_sites) for j in call.junctions] == [
        (3_200, 8_400, "neither"), (8_100, 4_000, "both")]
    five, middle, three = tables.partners("M000001")
    assert (middle.strand, middle.five_prime, middle.three_prime) == ("-", 8_400, 8_100)
    assert (middle.gene_start, middle.gene_end) == (8_000, 8_500)
    assert five.model == [(1_000, 1_300), (3_000, 3_200)]


def test_a_fusion_draws_from_the_tables_alone(fusion_tables, fusion_kwargs):
    view = FusionView("GENE5::GENEM::GENE3", fusion_table=fusion_tables,
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    view.plot()
    assert view.fusion_id == "M000001"
    assert [(g.name, len(g.reads), g.distinct) for g in view.groups] == [
        ("GENE5::GENEM::GENE3", 4, 0), ("GENEM::GENE3", 7, 2)]
    # Datasets fusion_Ctrl and fusion_Trt share the name up to the first underscore.
    assert view.condition_order == ["fusion"]
    assert all(r.polya_length is None for g in view.groups for r in g.reads)


def test_bams_group_the_datasets_and_give_the_tails(fusion_tables, fusion_kwargs):
    view = FusionView("M000001", fusion_table=fusion_tables,
                      bam_files=fusion_kwargs["bam_files"],
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    view.plot()
    assert view.condition_order == ["Ctrl", "Trt"]
    whole = view.groups[0]
    assert {r.condition for r in whole.reads} == {"Trt"}
    assert all(r.polya_length for r in whole.reads)


def test_tables_and_bams_draw_the_same_fusion(fusion_tables, fusion_kwargs):
    from_bams = FusionView(**fusion_kwargs)
    from_bams.prepare()
    from_tables = FusionView("M000001", fusion_table=fusion_tables,
                             bam_files=fusion_kwargs["bam_files"],
                             reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    from_tables.prepare()
    shape = lambda v: [(g.name, len(g.reads), g.distinct) for g in v.groups]  # noqa: E731
    assert shape(from_bams) == shape(from_tables)


def test_progress_names_the_join_a_partial_transcript_runs_past(fusion_kwargs, capsys):
    view = FusionView(**{**fusion_kwargs, "verbose": True})
    view.prepare()
    out = capsys.readouterr().out
    assert "GENE5::GENEM::GENE3: 4 reads\n" in out
    assert "GENEM::GENE3: 7 reads, 2 run past the GENE5 join" in out


def test_isoforge_transcripts_are_found_beside_the_tables(fusion_tables, fusion_tables_isoforms):
    from isotracks.fusion import FusionTables

    assert "isoforms" not in FusionTables(fusion_tables).paths
    for source in (fusion_tables_isoforms,
                   fusion_tables_isoforms + "_isoforge_fusion_isoforms.parquet"):
        tables = FusionTables(source)
        assert tables.paths["isoforms"].endswith("_isoforge_fusion_isoforms.parquet")
    found = tables.fusion_isoforms("M000001")
    assert sorted(found) == [1, 2, 3]
    assert found[2].parts == (1, 2) and not found[2].complete
    assert [found[n].isoform_id for n in (1, 2, 3)] == [
        "ISOFORF000000001", "ISOFORF000000002", "ISOFORF000000003"]
    assert found[2].blocks[2] == [(4_000, 4_450)]


def test_each_isoforge_transcript_is_a_pile_under_its_own_model(fusion_tables_isoforms, fusion_kwargs):
    view = FusionView("M000001", fusion_table=fusion_tables_isoforms,
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    view.plot()
    assert [(g.name, g.isoform, len(g.reads)) for g in view.groups] == [
        ("GENE5::GENEM::GENE3", 1, 4), ("GENEM::GENE3", 2, 5), ("GENEM::GENE3", 3, 2)]
    assert view.groups[1].model == {1: [(8_100, 8_480)], 2: [(4_000, 4_450)]}
    assert view.groups[2].model[2] == [(4_000, 4_600)]
    # Each called by IsoForge's ID for it, read from the tables.
    assert view.transcript_names == ["ISOFORF000000001", "ISOFORF000000002",
                                     "ISOFORF000000003"]
    assert view._heading(view.groups[1])[0].startswith("ISOFORF000000002 · GENEM::GENE3 · ")
    assert view._heading(view.groups[0])[0] == "ISOFORF000000001 · GENE5::GENEM::GENE3"


def test_progress_counts_reads_only_placed_with_a_transcript(fusion_tables_isoforms, fusion_kwargs, capsys):
    view = FusionView("M000001", fusion_table=fusion_tables_isoforms,
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=True)
    view.prepare()
    assert "ISOFORF000000003 (GENEM::GENE3): 2 reads, " in capsys.readouterr().out
    assert sum(r.fits is False for g in view.groups for r in g.reads) == 1


# ---------------------------------------------------------------- schematic

def _schematic(view):
    from isotracks.tracks import FusionSchematicTrack

    tracks = view.canvas.of_type(FusionSchematicTrack)
    return tracks[0] if tracks else None


def _gid_texts(track, gid):
    return [t.get_text() for t in track.ax.texts if t.get_gid() == gid]


def test_the_schematic_heads_the_figure_across_every_panel(drawn):
    schematic = _schematic(drawn)
    assert drawn.canvas.tracks[0] is schematic
    first = _panel_row(drawn)
    lefts = [first.ax] + [c.ax for c in first.panel_copies]
    box = schematic.ax.get_position()
    assert box.x0 == pytest.approx(lefts[0].get_position().x0)
    assert box.x1 == pytest.approx(lefts[-1].get_position().x1)
    assert not schematic.panel_copies, "one drawing, not a copy per panel"


def test_the_schematic_names_the_partners_and_their_breakpoints(drawn):
    schematic = _schematic(drawn)
    assert _gid_texts(schematic, "schematic-gene-name") == ["GENE5", "GENEM", "GENE3"]
    labels = [t.replace("\n", "") for t in _gid_texts(schematic, "schematic-coordinate")]
    assert labels == ["chrA:3,200", "chrB:8,400–8,100", "chrB:4,000"]


def test_the_fused_transcript_is_the_kept_exons_to_scale(drawn):
    schematic = _schematic(drawn)
    blocks = [p for p in schematic.ax.patches if p.get_gid() == "schematic-transcript"]
    widths = [p.get_width() for p in blocks]
    # GENE5 keeps 1,000-1,300 and 3,000-3,200, GENEM 8,100-8,400, GENE3 4,000-4,600.
    lengths = [301, 201, 301, 601]
    assert np.allclose(np.array(widths) / sum(widths), np.array(lengths) / sum(lengths))
    assert any("1,404 nt" in t.get_text() for t in schematic.ax.texts)


def test_a_minus_strand_gene_is_drawn_five_prime_first():
    from isotracks.tracks.fusion_schematic import gene_axis

    to_x = gene_axis([(8_000, 8_200), (8_300, 8_500)], "-")
    assert to_x(8_500.5) == pytest.approx(0.0) and to_x(7_999.5) == pytest.approx(1.0)
    assert to_x(8_400) < to_x(8_100)


def test_every_intron_is_drawn_the_same_width():
    from isotracks.tracks.fusion_schematic import gene_axis

    exons = [(1, 100), (1_001, 1_100), (500_001, 500_100)]
    to_x = gene_axis(exons, "+")
    short = to_x(1_000.5) - to_x(100.5)
    long = to_x(500_000.5) - to_x(1_100.5)
    assert short == pytest.approx(long)


def test_schematic_labels_stay_over_their_own_genes_and_clear_of_each_other(fusion_kwargs):
    theme.use(width=60)
    view = FusionView(**fusion_kwargs)
    figure = view.plot()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    schematic = _schematic(view)
    boxes = [t.get_window_extent(renderer) for t in schematic.ax.texts
             if t.get_gid() in ("schematic-coordinate", "schematic-gene-name")]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), "two schematic labels overlap"
    axes = schematic.ax.get_window_extent(renderer)
    every = [t.get_window_extent(renderer) for t in schematic.ax.texts]
    assert all(axes.x0 - 1 <= b.x0 and b.x1 <= axes.x1 + 1 for b in every)


def test_breakpoint_lines_stop_under_the_schematic(drawn):
    top = _schematic(drawn).ax.get_position().y0
    lines = [a for a in drawn.figure.artists if a.get_gid() == "fusion-breakpoint"]
    assert lines and all(max(line.get_ydata()) <= top + 1e-9 for line in lines)


def test_the_schematic_can_be_left_out(fusion_kwargs):
    view = FusionView(**{**fusion_kwargs, "schematic": False})
    view.plot()
    assert _schematic(view) is None


def test_the_schematic_draws_from_the_tables_without_a_gtf(fusion_tables):
    view = FusionView("M000001", fusion_table=fusion_tables, verbose=False)
    view.plot()
    genes = _schematic(view).config["genes"]
    assert [g.exons for g in genes] == [[], [], []]
    assert genes[1].kept == [(8_100, 8_400)]


def test_a_range_too_wide_for_its_gene_breaks_after_the_dash():
    from isotracks.canvas import Canvas
    from isotracks.tracks.fusion_schematic import SchematicGene

    genes = [
        SchematicGene("RAD23B", "chr9", "+", "#0072B2",
                      exons=[(107_283_279, 107_283_800), (107_290_000, 107_290_200)],
                      kept=[(107_283_255, 107_283_561)], donor=107_283_561),
        SchematicGene("RIMS2", "chr8", "+", "#D55E00",
                      exons=[(103_500_610, 103_501_062), (103_600_000, 103_600_200)],
                      kept=[(103_500_557, 103_501_062)],
                      acceptor=103_500_557, donor=103_501_062),
        SchematicGene("ATP6V1C1", "chr8", "+", "#009E73",
                      exons=[(103_021_083, 103_021_300), (103_062_955, 103_063_047)],
                      kept=[(103_062_955, 103_063_047)], acceptor=103_062_955),
    ]
    canvas = Canvas()
    canvas.add("FusionSchematicTrack", genes=genes)
    canvas.set_region(("chr8", 1, 1_000, "+"))
    figure = canvas.render(figsize=(3.5, 1.0))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    ax = canvas.tracks[0].ax
    coordinates = [t for t in ax.texts if t.get_gid() == "schematic-coordinate"]
    assert coordinates[1].get_text() == "chr8:103,500,557–\n103,501,062"
    boxes = [t.get_window_extent(renderer) for t in ax.texts
             if t.get_gid() in ("schematic-coordinate", "schematic-gene-name")]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), "two schematic labels overlap"


# ------------------------------------------------ the gene figure's structure

def _cluster_bed(path, rows):
    """A cluster BED in IsoForge's layout: (chrom, site, strand, motif or None)."""
    lines = []
    for chrom, site, strand, motif in rows:
        extra = [f"{site + 15}@{motif}", str(site)] if motif else ["."]
        lines.append("\t".join([chrom, str(site - 5), str(site + 5), str(site), "50", strand,
                                "exon", "G", "2.1", "50", "1.0", "3", *extra]))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_the_fusion_follows_the_gene_figure_top_to_bottom(drawn):
    kinds = [type(t).__name__ for t in drawn.canvas.tracks]
    assert kinds[:6] == ["FusionSchematicTrack", "HighlightLabelTrack", "EndDensityTrack",
                         "GeneModelTrack", "TranscriptTrack", "TranscriptTrack"]
    assert kinds[6] == "SpacerTrack" and "annotation" not in [t.label for t in drawn.canvas.tracks]


def test_the_gene_row_is_the_fusion_named_in_bold_with_its_transcripts_in_grey(drawn):
    from isotracks.tracks import GeneModelTrack

    gene = drawn.canvas.of_type(GeneModelTrack)[0]
    assert gene.label == "GENE5::\nGENEM::\nGENE3"  # wrapped to the gutter
    assert gene.config["label_weight"] == "bold" and gene.config["label_lines"] == 3
    # Only the whole fusion reaches GENE5, so this panel overlays one transcript.
    assert [f.id for f in gene.region_data] == ["GENE5"]
    assert all("facecolor" not in c.parm for f in gene.region_data for c in f.children)


def test_each_transcript_has_a_row_named_for_it(drawn):
    from isotracks.tracks import TranscriptTrack

    rows = [t for t in drawn.canvas.of_type(TranscriptTrack) if type(t) is TranscriptTrack]
    # No IsoForge IDs from a breakpoint row: the partners, broken to fit.
    assert [t.label for t in rows] == ["GENE5::\nGENEM::\nGENE3", "GENEM::\nGENE3"]


def test_a_long_fusion_name_breaks_only_between_genes():
    from isotracks.fusion_view import wrap_fusion_name

    assert wrap_fusion_name("RAD23B::RIMS2::ATP6V1C1", 9) == "RAD23B::\nRIMS2::\nATP6V1C1"
    assert wrap_fusion_name("RAD23B::RIMS2::ATP6V1C1", 15) == "RAD23B::RIMS2::\nATP6V1C1"
    assert wrap_fusion_name("RAD23B::RIMS2::ATP6V1C1", 40) == "RAD23B::RIMS2::ATP6V1C1"
    assert wrap_fusion_name("EML4::ALK", 3) == "EML4::\nALK"


def test_without_isoforge_ids_a_transcript_is_called_by_its_partners(
        fusion_tables_isoforms, fusion_kwargs, tmp_path):
    import shutil

    import pyarrow.parquet as pq

    source = os.path.dirname(fusion_tables_isoforms)
    for name in os.listdir(source):
        shutil.copy(os.path.join(source, name), tmp_path / name)
    for suffix in ("_isoforge_fusion_isoforms.parquet", "_isoforge_fusion_reads.parquet"):
        path = tmp_path / ("TEST" + suffix)
        table = pq.read_table(path)
        pq.write_table(table.drop_columns(["isoform_id"]), path)

    view = FusionView("M000001", fusion_table=str(tmp_path / "TEST"),
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    view.plot()
    assert view.transcript_names == ["GENE5::GENEM::GENE3", "GENEM::GENE3 (1 of 2)",
                                     "GENEM::GENE3 (2 of 2)"]
    assert _drawn_headings(view)[1].get_text().startswith("GENEM::GENE3 (1 of 2) \u00b7 ")


def test_read_ends_are_where_each_read_began_and_ended(drawn):
    from isotracks.fusion import read_ends

    reads = [r for g in drawn.groups for r in g.reads]
    fives, threes = read_ends(drawn.partners, reads)
    assert len(fives["chrA"]) == 4 and all(v <= 1_300 for v in fives["chrA"])
    assert len(fives["chrB"]) == 7 and all(v >= 8_400 for v in fives["chrB"])
    assert len(threes["chrB"]) == 11 and "chrA" not in threes


def test_read_ends_share_one_scale_across_panels(drawn):
    from isotracks.tracks import EndDensityTrack

    track = drawn.canvas.of_type(EndDensityTrack)[0]
    peaks = [t.region_data.peak for t in [track] + track.panel_copies]
    top = track.config["fix_ylim"][1]
    assert top == pytest.approx(max(peaks)) and top > peaks[0]
    assert all(t.ylim[1] >= top for t in [track] + track.panel_copies)
    assert track.config["sublabel"] == f"0–{track._format_scale(top)}"


def test_sites_are_drawn_at_the_transcripts_ends_in_their_own_panels(fusion_kwargs, tmp_path):
    from isotracks.tracks import SiteTrack

    tss = _cluster_bed(tmp_path / "tss.bed", [
        ("chrA", 1_000, "+", None),  # the whole fusion's start, in GENE5
        ("chrB", 8_400, "-", None),  # the partial one's, in GENEM
        ("chrB", 8_402, "+", None),  # the wrong strand for GENEM
        ("chrA", 3_100, "+", None),  # no transcript starts here
    ])
    polya = _cluster_bed(tmp_path / "polya.bed", [
        ("chrB", 4_600, "+", "AATAAA"),  # where both end, in GENE3
        ("chrA", 4_600, "+", "AATAAA"),  # the same coordinate on another chromosome
    ])
    view = FusionView(**{**fusion_kwargs, "tss": tss, "polya": polya})
    view.plot()
    polya_track, tss_track = view.canvas.of_type(SiteTrack)
    assert (polya_track.label, tss_track.label) == ("poly(A)", "TSS")

    def per_panel(track):
        return [sorted(zip(t.region_data["chrom"], t.region_data["site"]))
                for t in [track] + track.panel_copies]

    assert per_panel(tss_track) == [[("chrA", 1_000)], [("chrB", 8_400)], []]
    assert per_panel(polya_track) == [[], [], [("chrB", 4_600)]]


def test_site_frames_hand_each_panel_only_its_own_chromosome():
    import pandas as pd

    from isotracks.io.clusters import SITE_COLUMNS, SiteFrameReader

    frame = pd.DataFrame([["chrA", 0, 0, 4_000, 1, "+", "", "", None, None],
                          ["chrB", 0, 0, 4_000, 1, "+", "", "", None, None]],
                         columns=SITE_COLUMNS)
    found = SiteFrameReader(frame).read_region(("chrB", 4_700, 3_900, "-"))
    assert list(found["chrom"]) == ["chrB"]


# ------------------------------------------------- joins across the panel gaps

def _joins(view):
    """The intron joins (as boxes) and direct joins (as segments), in figure units."""
    bands = [a for a in view.figure.artists if a.get_gid() == "fusion-join-intron"]
    dots = [a for a in view.figure.artists if a.get_gid() == "fusion-join-direct"]
    boxes = [p.vertices[:4] for p in bands[0].get_paths()] if bands else []
    return boxes, (dots[0].get_segments() if dots else [])


def test_a_spliced_junction_is_drawn_as_an_intron_and_a_direct_one_dotted(drawn):
    assert [drawn.spliced(j) for j in (0, 1)] == [False, True]
    boxes, dots = _joins(drawn)
    assert len(dots) == 1 + 1 + 4
    assert len(boxes) == 1 + 1 + 1 + 4 + 7


def test_a_join_meets_each_read_where_it_was_cut(drawn):
    to_figure = drawn.figure.transFigure.inverted()
    boxes, dots = _joins(drawn)
    trt = _stacks(drawn, 0)[1]  # the whole fusion's 4 reads
    left, right = trt, trt.panel_copies[0]
    ends = {f.read_name: f for f in right.region_data}
    expected = []
    for feature in left.region_data:
        other = ends[feature.read_name]
        bottom, height, _ = left.intron_band(feature)
        x0, y = to_figure.transform(left.ax.transData.transform(
            (feature.end + 0.5, bottom + height / 2)))
        x1 = to_figure.transform(right.ax.transData.transform((other.end + 0.5, 0)))[0]
        expected.append((x0, x1, y))
    drawn_joins = [(a[0], b[0], a[1]) for a, b in dots]
    for x0, x1, y in expected:
        assert x0 < x1
        assert any(np.allclose((x0, x1, y), j) for j in drawn_joins)

    # The second junction's boxes are as tall as the reads' intron band.
    ctrl = _stacks(drawn, 1)[0]
    middle = ctrl.panel_copies[0]  # GENEM, where the join starts
    heights = sorted(round(b[2][1] - b[0][1], 9) for b in boxes)
    for feature in middle.region_data:
        bottom, height, colour = middle.intron_band(feature)
        (_, low), (_, high) = to_figure.transform(middle.ax.transData.transform(
            [(0, bottom), (0, bottom + height)]))
        assert round(high - low, 9) in heights


def test_isoforge_ids_fit_their_gutter(fusion_tables_isoforms, fusion_kwargs):
    view = FusionView("M000001", fusion_table=fusion_tables_isoforms,
                      reference_gtf=fusion_kwargs["reference_gtf"], verbose=False)
    figure = view.plot()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = [text for track in view.canvas.tracks for text in track.gutter_ax.texts]
    assert any(t.get_text() == "ISOFORF000000001" for t in labels)
    assert all(t.get_window_extent(renderer).x0 >= figure.bbox.x0 for t in labels)



def test_datasets_are_grouped_by_name_or_as_given(fusion_tables):
    by_name = FusionView("M000001", fusion_table=fusion_tables, verbose=False)
    by_name.prepare()
    assert {r.condition for g in by_name.groups for r in g.reads} == {"fusion"}

    given = FusionView("M000001", fusion_table=fusion_tables, verbose=False,
                       conditions={"Ctrl": ["fusion_Ctrl"], "Trt": ["fusion_Trt"]})
    given.prepare()
    assert given.condition_order == ["Ctrl", "Trt"]
