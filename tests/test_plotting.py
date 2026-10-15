"""End-to-end rendering, isoform selection, zoom and the poly(A) table."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from isotracks import IsoTracks, Theme, plot_gene
from isotracks.browser import BamInputError
from isotracks.tracks import CoverageTrack, ReadTrack, TranscriptTrack

from .conftest import GENE, ISOFORMS


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_locus_comes_from_the_transcripts_not_the_gene_feature(view_kwargs):
    locus = IsoTracks(**view_kwargs).locus
    assert (locus.start, locus.end) == (1_000, 6_800)
    assert locus.source == "transcripts"


def test_both_isoforms_pass_the_default_threshold(view_kwargs):
    assert IsoTracks(**view_kwargs).transcripts() == sorted(ISOFORMS)


def test_min_reads_drops_everything_when_set_high(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "min_reads": 1000})
    assert view.transcripts() == []


def test_per_condition_filter_requires_reads_in_every_arm(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "min_reads_per_condition": 12})
    assert view.transcripts() == sorted(ISOFORMS)  # 12 per condition exactly
    strict = IsoTracks(**{**view_kwargs, "min_reads_per_condition": 13})
    assert strict.transcripts() == []


def test_max_isoforms_keeps_the_best_supported(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1})
    kept = view.transcripts()
    assert len(kept) == 1
    counts = view.gene_reads.counts()
    assert counts[kept[0]] == max(counts.values())


def test_max_isoforms_none_draws_everything(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": None})
    assert view.transcripts() == sorted(ISOFORMS)


def test_the_cap_says_what_it_dropped(view_kwargs, capsys):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1, "verbose": True})
    view.transcripts()
    assert "drawing the 1 best supported" in capsys.readouterr().out


def test_every_read_stack_is_the_same_height(view_kwargs):
    from isotracks import Layout

    view = IsoTracks(**view_kwargs)
    canvas = view.build()
    heights = {t.config["height"] for t in canvas.of_type(ReadTrack)}
    assert heights == {view.layout.read_stack}
    assert len(canvas.of_type(ReadTrack)) > 1, "nothing was compared"

    roomy = IsoTracks(**{**view_kwargs, "compact": False}).build()
    assert {t.config["height"] for t in roomy.of_type(ReadTrack)} == {
        Layout().read_stack}


def test_stack_height_is_configurable(view_kwargs):
    from isotracks import Layout

    view = IsoTracks(**{**view_kwargs, "layout": Layout(read_stack=5.5)})
    canvas = view.build()
    assert {t.config["height"] for t in canvas.of_type(ReadTrack)} == {5.5}


def test_a_stack_states_its_read_count(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot()
    stack = view.canvas.of_type(ReadTrack)[0]
    assert stack.config["sublabel"] == f"n = {stack.read_count:,}"
    gutter = [t.get_text() for t in stack.gutter_ax.texts]
    assert stack.config["sublabel"] in gutter  # in the gutter, not over the reads


def test_a_thinned_stack_reports_both_numbers(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=2)
    stack = view.canvas.of_type(ReadTrack)[0]
    assert stack.config["sublabel"] == "n = 2/12"


def test_the_count_never_covers_the_reads(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot()
    stack = view.canvas.of_type(ReadTrack)[0]
    assert len(stack.ax.texts) == 0


def test_isoform_colors_are_distinct(view_kwargs):
    view = IsoTracks(**view_kwargs)
    colors = {view.isoform_color(tid) for tid in view.transcripts()}
    assert len(colors) == len(ISOFORMS)


def test_transcripts_are_stable_across_repeated_calls(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "min_reads_per_condition": 5})
    first = view.transcripts()
    assert view.transcripts(refresh=True) == first
    assert view.transcripts(refresh=True) == first


def test_display_region_leaves_room_for_the_longest_tail(view_kwargs):
    view = IsoTracks(**view_kwargs)
    _, _, end, _ = view.display_region(padding="auto")
    # Longest synthetic tail is 180 + 5*10 = 230 nt past the 6,800 bp end.
    assert end >= 6_800 + 230


def test_explicit_padding_is_respected(view_kwargs):
    _, _, end, _ = IsoTracks(**view_kwargs).display_region(padding=1_000)
    assert end == 6_800 + 1_000


def test_polya_sites_are_matched_to_the_drawn_isoforms(view_kwargs):
    sites = IsoTracks(**view_kwargs).polya_sites()
    assert sorted(sites["site"]) == [3_600, 6_800]
    assert set(sites["motif"]) == {"AATAAA", "ATTAAA"}


def test_tss_sites_are_matched_to_the_drawn_isoforms(view_kwargs):
    sites = IsoTracks(**view_kwargs).tss_sites()
    assert list(sites["site"]) == [1_000]


def test_both_end_tracks_are_drawn(view_kwargs):
    from isotracks.tracks import SiteTrack

    canvas = IsoTracks(**view_kwargs).build()
    labels = [t.label for t in canvas.of_type(SiteTrack)]
    # poly(A) on top.
    assert labels == ["poly(A)", "TSS"]


def test_build_produces_the_expected_track_stack(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**view_kwargs).build()
    # GeneModelTrack is a TranscriptTrack, so of_type() matches it too.
    models = [t for t in canvas.of_type(TranscriptTrack)
              if not isinstance(t, GeneModelTrack)]
    assert len(models) == 2  # one per isoform
    assert len(canvas.of_type(GeneModelTrack)) == 1  # one per figure
    assert len(canvas.of_type(ReadTrack)) == 4  # 2 isoforms x 2 conditions
    assert len(canvas.of_type(CoverageTrack)) == 2  # one per condition
    assert canvas.tracks[-1].__class__.__name__ == "CoordinateAxisTrack"


def test_plot_renders_and_writes_a_pdf(view_kwargs, tmp_path):
    output = tmp_path / "gene.pdf"
    figure = IsoTracks(**view_kwargs).plot(output=str(output))
    assert output.exists() and output.stat().st_size > 1_000
    assert figure.get_size_inches()[0] == pytest.approx(89 / 25.4, rel=1e-3)


def test_a_stem_writes_both_a_vector_and_a_raster_copy(view_kwargs, tmp_path):
    IsoTracks(**view_kwargs).plot(output=str(tmp_path / "gene"))
    assert (tmp_path / "gene.pdf").exists()
    assert (tmp_path / "gene.png").exists()


def test_an_explicit_extension_writes_only_that(view_kwargs, tmp_path):
    IsoTracks(**view_kwargs).plot(output=str(tmp_path / "gene.svg"))
    assert (tmp_path / "gene.svg").exists()
    assert not (tmp_path / "gene.png").exists()


def test_formats_override_the_default_pair(view_kwargs, tmp_path):
    IsoTracks(**view_kwargs).plot(output=str(tmp_path / "gene"), formats=("png",))
    assert (tmp_path / "gene.png").exists()
    assert not (tmp_path / "gene.pdf").exists()


def test_polya_table_comes_from_the_annotation_when_it_has_lengths(view_kwargs, tmp_path):
    import pandas as pd

    annot = pd.read_csv(view_kwargs["read_annot"], sep="\t")
    annot["polya_length"] = 111
    path = tmp_path / "with_lengths.tsv"
    annot.to_csv(path, sep="\t", index=False)

    view = IsoTracks(**{**view_kwargs, "read_annot": str(path)})
    table = view.polya_table()
    assert set(table["polya_length"]) == {111}
    assert set(table["condition"]) == {"Ctrl", "Trt"}


def test_polya_table_can_be_forced_to_the_bam_tags(view_kwargs, tmp_path):
    import pandas as pd

    annot = pd.read_csv(view_kwargs["read_annot"], sep="\t")
    annot["polya_length"] = 111
    path = tmp_path / "with_lengths.tsv"
    annot.to_csv(path, sep="\t", index=False)

    view = IsoTracks(**{**view_kwargs, "read_annot": str(path)})
    table = view.polya_table(source="bam")
    assert 111 not in set(table["polya_length"])  # the Dorado tags, not the table


def test_polya_table_rejects_an_unknown_source(view_kwargs):
    with pytest.raises(ValueError, match="unknown source"):
        IsoTracks(**view_kwargs).polya_table(source="guess")


def test_plot_gene_convenience_matches_the_class(view_kwargs, tmp_path):
    kwargs = dict(view_kwargs)
    kwargs.pop("gene_name")
    figure = plot_gene(gene_name=GENE, output=str(tmp_path / "g.png"), **kwargs)
    assert figure is not None


def test_diamond_ordering_places_the_longest_tail_in_the_middle(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(sort_style="diamond")
    stack = view.canvas.of_type(ReadTrack)[0]
    frame = stack.region_data.df
    middle_row = frame.loc[frame["polya_length"].idxmax(), "order"]
    assert middle_row == len(frame) // 2


def test_waterfall_ordering_puts_the_longest_tail_first(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(sort_style="waterfall")
    frame = view.canvas.of_type(ReadTrack)[0].region_data.df
    assert frame.loc[frame["order"] == 0, "polya_length"].iloc[0] == frame["polya_length"].max()


def test_reads_carry_a_polya_tail_feature(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot()
    stack = view.canvas.of_type(ReadTrack)[0]
    tails = [
        child for feature in stack.region_data for child in feature.children
        if child.type == "polyA"
    ]
    assert tails, "reads with a pt tag must gain a poly(A) feature"


def test_tail_extends_the_read_in_the_layout_frame(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot()
    stack = view.canvas.of_type(ReadTrack)[0]
    frame = stack.region_data.df
    for feature in stack.region_data:
        row = frame[frame["id"] == feature.id]
        if not row.empty:
            assert int(row["end"].iloc[0]) == feature.end


def test_shared_coverage_scale_puts_conditions_on_one_axis(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(share_coverage_scale=True)
    limits = {track.ylim for track in view.canvas.of_type(CoverageTrack)}
    assert len(limits) == 1


def test_max_reads_thins_the_stack(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=3)
    for stack in view.canvas.of_type(ReadTrack):
        assert stack.read_count <= 3


def test_thinning_keeps_the_tail_length_range(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=3)
    frame = view.canvas.of_type(ReadTrack)[0].region_data.df
    assert frame["polya_length"].max() == 60 + 5 * 10  # longest kept
    assert frame["polya_length"].min() == 60  # shortest kept


def test_thinning_does_not_touch_the_polya_table(view_kwargs):
    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=2)
    assert len(view.polya_table()) == 48


def test_zoom_to_last_exon_covers_the_three_prime_end(view_kwargs):
    view = IsoTracks(**view_kwargs)
    chrom, start, end, strand = view.zoom(
        "last_exon", transcript_id="TX_LONG", padding=100, tail_room=False)
    assert (start, end, strand) == (6_000 - 100, 6_800 + 100, "+")


def test_zooming_to_the_three_prime_end_leaves_room_for_the_tails(view_kwargs):
    view = IsoTracks(**view_kwargs)
    _, _, clipped, _ = view.zoom("last_exon", transcript_id="TX_LONG",
                                 padding=100, tail_room=False)
    _, _, roomy, _ = view.zoom("last_exon", transcript_id="TX_LONG", padding=100)
    assert roomy >= clipped + 230  # the longest synthetic tail


def test_zoom_region_is_what_gets_drawn(view_kwargs):
    view = IsoTracks(**view_kwargs)
    window = view.zoom("first_exon", transcript_id="TX_LONG", padding=50)
    view.plot(region=window)
    assert view.canvas.region == window


def test_zoom_rejects_an_out_of_range_exon(view_kwargs):
    view = IsoTracks(**view_kwargs)
    with pytest.raises(ValueError, match="exon 9 does not exist"):
        view.zoom("exon_9", transcript_id="TX_SHORT")


def test_zoom_rejects_an_unknown_target(view_kwargs):
    with pytest.raises(ValueError, match="unknown zoom target"):
        IsoTracks(**view_kwargs).zoom("middle_bit")


def test_exons_are_numbered_and_measured_inclusively(view_kwargs):
    exons = IsoTracks(**view_kwargs).exons("TX_LONG")
    assert list(exons["exon_number"]) == [1, 2, 3]
    assert list(exons["length"]) == [301, 201, 801]  # inclusive GTF spans
    assert list(exons["position"]) == ["5_prime", "internal", "3_prime"]


def test_polya_table_reports_every_read(view_kwargs):
    table = IsoTracks(**view_kwargs).polya_table()
    assert len(table) == 48
    assert set(table["condition"]) == {"Ctrl", "Trt"}
    assert table.groupby("condition")["polya_length"].mean()["Trt"] > \
        table.groupby("condition")["polya_length"].mean()["Ctrl"]


def test_a_view_needs_a_gene_or_a_region():
    with pytest.raises(ValueError, match="gene_name or region"):
        IsoTracks()


def test_missing_bam_is_reported_clearly(view_kwargs):
    with pytest.raises(BamInputError, match="not found"):
        IsoTracks(**{**view_kwargs, "bam_files": {"Ctrl": "/nowhere/missing.bam"}})


def test_unindexed_bam_is_reported_clearly(view_kwargs, tmp_path, bam_files):
    import shutil
    orphan = tmp_path / "unindexed.bam"
    shutil.copy(bam_files["Ctrl"][0], orphan)
    with pytest.raises(BamInputError, match="samtools index"):
        IsoTracks(**{**view_kwargs, "bam_files": {"Ctrl": str(orphan)}})


def test_a_gene_name_without_a_gtf_is_refused(bam_files):
    with pytest.raises(ValueError, match="needs a GTF"):
        IsoTracks(gene_name=GENE, bam_files=bam_files, verbose=False).locus


def test_unknown_gene_names_the_files_searched(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "gene_name": "NO_SUCH_GENE"})
    with pytest.raises(ValueError, match="not found in"):
        _ = view.locus


def test_region_only_view_draws_every_read(view_kwargs, tmp_path):
    kwargs = {k: v for k, v in view_kwargs.items() if k not in ("read_annot", "bed_file")}
    view = IsoTracks(**{**kwargs, "region": ("chrT", 900, 7_000, "+")})
    view.plot(output=str(tmp_path / "region.png"), padding=300)
    assert len(view.canvas.of_type(ReadTrack)) == 2


def test_width_changes_the_figure_but_not_the_type_size(view_kwargs):
    narrow = IsoTracks(**view_kwargs).plot(width="single")
    wide = IsoTracks(**view_kwargs).plot(width="double")
    assert wide.get_size_inches()[0] > narrow.get_size_inches()[0]
    assert Theme(width="single").fonts == Theme(width="double").fonts


def _part_centroids(tmp_path, bed_line, dpi=200):
    """Ink-weighted vertical centroid of a model's parts, in rendered pixels."""
    import numpy as np
    from PIL import Image

    from isotracks import Canvas

    bed = tmp_path / "model.bed"
    bed.write_text(bed_line + "\n")

    canvas = Canvas()
    track = canvas.add("TranscriptTrack", file_name=str(bed), label="")
    canvas.add("CoordinateAxisTrack")
    figure = canvas.render(region=("chrT", 900, 5100, "+"), figsize=(4, 1.2))
    output = tmp_path / "model.png"
    figure.savefig(output, dpi=dpi)  # no tight bbox: keeps the mapping

    box = track.ax.get_window_extent()
    scale = dpi / figure.dpi
    height = int(figure.get_figheight() * dpi)
    crop = np.asarray(Image.open(output).convert("L")).astype(float)[
        height - int(box.y1 * scale):height - int(box.y0 * scale),
        int(box.x0 * scale):int(box.x1 * scale),
    ]
    darkness = 255.0 - crop
    rows = np.arange(crop.shape[0])

    centroids = {}
    # Columns chosen to land inside each part of the model built below.
    for name, fraction in (("intron", 0.32), ("5UTR", 0.10),
                           ("CDS", 0.44), ("3UTR", 0.93)):
        column = darkness[:, int(crop.shape[1] * fraction)]
        if column.sum() > 0:
            centroids[name] = float((rows * column).sum() / column.sum())
    return centroids


def test_every_part_of_a_model_sits_on_one_midline(tmp_path):
    centroids = _part_centroids(tmp_path, "\t".join([
        "chrT", "1000", "5000", "TX", "0", "+", "2000", "4000",
        "0", "3", "500,500,500,", "0,1500,3500,",
    ]))
    assert set(centroids) == {"intron", "5UTR", "CDS", "3UTR"}
    spread = max(centroids.values()) - min(centroids.values())
    assert spread < 0.25, f"model parts are off one another by {spread:.2f} px: {centroids}"


def test_a_utr_is_shorter_than_the_coding_block_and_hollow(tmp_path):
    from isotracks import Canvas
    from isotracks.tracks.features import UTR_TYPES

    bed = tmp_path / "model.bed"
    bed.write_text("\t".join([
        "chrT", "1000", "5000", "TX", "0", "+", "2000", "4000",
        "0", "3", "500,500,500,", "0,1500,3500,",
    ]) + "\n")
    canvas = Canvas()
    track = canvas.add("TranscriptTrack", file_name=str(bed), label="")
    canvas.add("CoordinateAxisTrack")
    canvas.render(region=("chrT", 900, 5100, "+"), figsize=(4, 1.2))

    assert track.config["thin_height"] < track.config["body_height"]
    assert track.config["hollow_utr"]
    assert UTR_TYPES == {"5UTR", "3UTR"}


#: Small exons close together, then one far downstream.
SMALL_EXONS = [(1_000, 1_064), (1_364, 1_428), (1_728, 1_792), (200_000, 200_100)]


def _small_exon_track(tmp_path, region, intron_map=None, **config):
    """A model of :data:`SMALL_EXONS` drawn across *region*."""
    from isotracks import Canvas

    first, last = SMALL_EXONS[0][0], SMALL_EXONS[-1][1]
    bed = tmp_path / "long.bed"
    bed.write_text("\t".join([
        "chrT", str(first - 1), str(last), "LONG", "0", "+", str(first - 1), str(last), "0",
        str(len(SMALL_EXONS)),
        ",".join(str(e - s + 1) for s, e in SMALL_EXONS) + ",",
        ",".join(str(s - first) for s, _ in SMALL_EXONS) + ",",
    ]) + "\n")
    canvas = Canvas()
    track = canvas.add("TranscriptTrack", file_name=str(bed), label="", **config)
    canvas.add("CoordinateAxisTrack")
    canvas.set_intron_map(intron_map)
    canvas.render(region=region, figsize=(4, 1.2))
    return track


def _body_widths_pt(track):
    """Printed width of every exon block the track drew, left to right."""
    blocks = []
    for collection in track.ax.collections:
        if collection.get_zorder() == track.config["style"]["intron"]["zorder"]:
            continue
        for path in collection.get_paths():
            xs = track.ax.transData.transform(path.vertices)[:, 0]
            blocks.append((xs.min(), (xs.max() - xs.min()) * 72.0 / track.ax.figure.dpi))
    return [width for _, width in sorted(blocks)]


def test_a_small_exon_far_from_its_neighbours_is_still_drawn(tmp_path):
    track = _small_exon_track(tmp_path, ("chrT", 0, 220_000, "+"))
    widths = _body_widths_pt(track)
    floor = track.config["min_body_width_pt"] * track.theme.stroke
    assert len(widths) == len(SMALL_EXONS)
    assert min(widths) == pytest.approx(floor, rel=0.02), widths


def test_widening_an_intron_never_eats_the_exons_beside_it(tmp_path):
    track = _small_exon_track(tmp_path, ("chrT", 0, 220_000, "+"))
    from isotracks.tracks.features import BODY_TYPES

    model = next(iter(track.region_data))
    exons = [c for c in track._spaced_children(model) if c.type in BODY_TYPES]
    assert len(exons) == len(SMALL_EXONS)
    for exon, (start, end) in zip(exons, SMALL_EXONS):
        kept = min(end - start + 1, track._min_body)
        assert exon.end - exon.start + 1 >= kept - 1e-9


def test_exons_wide_enough_to_see_keep_their_true_width(tmp_path):
    track = _small_exon_track(tmp_path, ("chrT", 900, 1_900, "+"))
    span_pt = track.ax.get_window_extent().width * 72.0 / track.ax.figure.dpi
    true_pt = 65 / 1_000 * span_pt
    assert _body_widths_pt(track)[0] == pytest.approx(true_pt, rel=0.02)


def test_the_minimum_exon_width_can_be_turned_off(tmp_path):
    track = _small_exon_track(tmp_path, ("chrT", 0, 220_000, "+"), min_body_width_pt=0,
                              min_intron_fraction=0)
    span_pt = track.ax.get_window_extent().width * 72.0 / track.ax.figure.dpi
    assert _body_widths_pt(track)[0] == pytest.approx(65 / 220_000 * span_pt, rel=0.02)


def test_small_exons_are_drawn_when_introns_are_shortened(tmp_path):
    from isotracks.introns import build_intron_map

    region = ("chrT", 0, 400_000, "+")
    track = _small_exon_track(tmp_path, region, build_intron_map(SMALL_EXONS, region, "scaled"))
    floor = track.config["min_body_width_pt"] * track.theme.stroke
    widths = _body_widths_pt(track)
    assert len(widths) == len(SMALL_EXONS)
    assert min(widths) == pytest.approx(floor, rel=0.02), widths


def test_end_density_sits_between_the_clusters_and_the_models(view_kwargs):
    from isotracks.tracks import EndDensityTrack, SiteTrack, TranscriptTrack

    canvas = IsoTracks(**view_kwargs).build()
    kinds = [type(t) for t in canvas.tracks]
    density = kinds.index(EndDensityTrack)
    assert density > max(i for i, k in enumerate(kinds) if k is SiteTrack)
    assert density < min(i for i, k in enumerate(kinds) if k is TranscriptTrack)


def test_end_density_can_be_turned_off(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    canvas = IsoTracks(**{**view_kwargs, "end_density": False}).build()
    assert canvas.of_type(EndDensityTrack) == []


def test_end_density_pools_the_conditions(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    tracks = view.canvas.of_type(EndDensityTrack)
    assert len(tracks) == 1
    assert tracks[0].region_data.five_prime_reads == 48  # both conditions


def test_both_end_curves_share_one_scale(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    track = view.canvas.of_type(EndDensityTrack)[0]
    assert track.data_ylim[1] == pytest.approx(track.region_data.peak)


def test_end_density_reports_its_peak_in_the_gutter(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    track = view.canvas.of_type(EndDensityTrack)[0]
    assert track.config["sublabel"].startswith("0–")
    assert track.config["sublabel"] in [t.get_text() for t in track.gutter_ax.texts]


def test_the_end_curves_are_unlabelled_by_default(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    track = view.canvas.of_type(EndDensityTrack)[0]
    assert track.ax.texts == [] or len(track.ax.texts) == 0


def test_the_end_curves_can_be_labelled_for_standalone_use(view_kwargs):
    from isotracks.tracks import EndDensityTrack

    view = IsoTracks(**view_kwargs)
    canvas = view.build()
    canvas.configure_all({"label_curves": True}, EndDensityTrack)
    canvas.set_region(view.display_region())
    canvas.render(figsize=(4, 8))
    labels = [t.get_text() for t in canvas.of_type(EndDensityTrack)[0].ax.texts]
    assert "5′ ends" in labels and "3′ ends" in labels


# ------------------------------------------------------------ modifications

def test_no_modification_track_without_a_pileup(view_kwargs):
    from isotracks.tracks import ModificationTrack

    assert IsoTracks(**view_kwargs).build().of_type(ModificationTrack) == []


def test_a_modification_track_per_condition_after_the_reads(view_kwargs, modkit_pileups):
    from isotracks.tracks import CoverageTrack, ModificationTrack, ReadTrack

    canvas = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups}).build()
    mods = canvas.of_type(ModificationTrack)
    assert [t.label for t in mods] == ["Ctrl", "Trt"]

    kinds = [type(t) for t in canvas.tracks]
    first_mod = kinds.index(ModificationTrack)
    assert first_mod > max(i for i, k in enumerate(kinds) if k is ReadTrack)
    assert first_mod < min(i for i, k in enumerate(kinds) if k is CoverageTrack)


def test_only_the_first_modification_track_carries_the_key(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    canvas = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups}).build()
    keys = [t.config["show_key"] for t in canvas.of_type(ModificationTrack)]
    assert keys == [True, False]


def test_the_modification_axis_is_always_a_full_fraction(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    view.plot(region=("chrT", 6_000, 6_600, "+"))
    for track in view.canvas.of_type(ModificationTrack):
        assert track.data_ylim == (0.0, 1.0)


def test_the_gutter_reports_sites_drawn_against_sites_assessed(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    view.plot(region=("chrT", 6_000, 6_600, "+"))
    labels = [t.config["sublabel"] for t in view.canvas.of_type(ModificationTrack)]
    assert all("/" in label and label.endswith("sites") for label in labels)


def test_the_treated_arm_shows_the_site_the_control_lacks(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    view.plot(region=("chrT", 6_000, 6_600, "+"))
    by_label = {t.label: t.region_data for t in view.canvas.of_type(ModificationTrack)}
    assert 6_100 not in set(by_label["Ctrl"].frame["position"])  # 10% modified
    assert 6_100 in set(by_label["Trt"].frame["position"])  # 80% modified


def test_too_many_mod_codes_are_capped_at_what_colour_can_carry(view_kwargs, capsys):
    from isotracks.theme import MAX_OVERLAID_MODS

    view = IsoTracks(**{**view_kwargs, "verbose": True,
                        "mod_codes": ("a", "17802", "m", "h", "17596")})
    assert len(view.mod_codes) == MAX_OVERLAID_MODS
    assert "shows at most" in capsys.readouterr().out


def test_mod_colours_are_handed_out_in_order(view_kwargs):
    from isotracks import Theme

    palette = Theme().mod_palette(["a", "17802"])
    assert len(set(palette.values())) == 2


def test_a_flat_list_of_codes_shares_one_track(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                        "mod_codes": ("a", "17802")})
    assert view.mod_groups == [["a", "17802"]]
    tracks = view.build().of_type(ModificationTrack)
    assert len(tracks) == 2  # one per condition, not per code


def test_nested_codes_get_a_track_each(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                        "mod_codes": [("a",), ("17802",)]})
    assert view.mod_groups == [["a"], ["17802"]]
    tracks = view.build().of_type(ModificationTrack)
    assert len(tracks) == 4  # 2 groups x 2 conditions
    assert [t.reader.codes for t in tracks] == [["a"], ["a"], ["17802"], ["17802"]]


def test_groups_can_mix_overlaid_and_separate(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                        "mod_codes": [("a", "17802"), ("m",)]})
    tracks = view.build().of_type(ModificationTrack)
    assert [t.reader.codes for t in tracks[:2]] == [["a", "17802"]] * 2
    assert [t.reader.codes for t in tracks[2:]] == [["m"]] * 2


def test_each_group_names_its_codes_once(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                        "mod_codes": [("a",), ("17802",)]})
    keys = [t.config["show_key"] for t in view.build().of_type(ModificationTrack)]
    assert keys == [True, False, True, False]


def test_a_code_keeps_its_colour_across_groups(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    view = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                        "mod_codes": [("a", "17802"), ("a",)]})
    palettes = [t.config["colors"] for t in view.build().of_type(ModificationTrack)]
    assert len({p["a"] for p in palettes}) == 1


def test_a_group_is_capped_at_what_colour_can_carry(view_kwargs, capsys):
    from isotracks.theme import MAX_OVERLAID_MODS

    view = IsoTracks(**{**view_kwargs, "verbose": True,
                        "mod_codes": [("a", "17802", "m", "h", "17596")]})
    assert view.mod_groups == [["a", "17802", "m"][:MAX_OVERLAID_MODS]]
    assert "own group" in capsys.readouterr().out


def test_mod_codes_reports_every_code_across_groups(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "mod_codes": [("a", "17802"), ("a", "m")]})
    assert view.mod_codes == ["a", "17802", "m"]  # de-duplicated, in order


# ------------------------------------------------------------ sashimi arcs

def _arc_curves(track):
    """The junction arcs a coverage track drew, as (x, y) arrays."""
    return [line.get_data() for line in track.ax.lines if len(line.get_xdata()) > 50]


def test_every_junction_arc_stays_inside_the_axes(view_kwargs):
    from isotracks.tracks import CoverageTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    for track in view.canvas.of_type(CoverageTrack):
        top = track.ax.get_ylim()[1]
        for _x, y in _arc_curves(track):
            assert max(y) <= top + 1e-6, f"arc peaks at {max(y):.1f} above ylim {top:.1f}"


def test_every_junction_gets_its_own_arc(view_kwargs):
    from isotracks.tracks import CoverageTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    for track in view.canvas.of_type(CoverageTrack):
        junctions = track.region_data[1]
        drawn = len(_arc_curves(track))
        assert drawn == len(junctions), f"{len(junctions)} junctions, {drawn} arcs"


def test_arcs_of_similar_span_get_different_heights(view_kwargs):
    from isotracks.tracks import CoverageTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    track = view.canvas.of_type(CoverageTrack)[0]
    apexes = [round(float(max(y)), 3) for _x, y in _arc_curves(track)]
    assert len(set(apexes)) == len(apexes), f"arcs share apex heights: {apexes}"


def test_coverage_uses_only_the_filtered_reads(view_kwargs):
    from isotracks.tracks import CoverageTrack

    filtered = IsoTracks(**view_kwargs)
    filtered.plot()
    peak = filtered.canvas.of_type(CoverageTrack)[0].peak_depth()

    track = filtered.canvas.of_type(CoverageTrack)[0]
    track.reader.read_filter = None  # what no filter would have given
    track.load_region_data()
    assert track.peak_depth() >= peak


# ------------------------------------------------ differential modification

def test_a_dmr_track_is_drawn_after_the_modification_tracks(view_kwargs, modkit_pileups,
                                                            mod_diff_table):
    from isotracks.tracks import CoverageTrack, ModDiffTrack, ModificationTrack

    canvas = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                          "mod_diff": {"Ctrl vs Trt": mod_diff_table}}).build()
    kinds = [type(t) for t in canvas.tracks]
    diff = kinds.index(ModDiffTrack)
    assert diff > max(i for i, k in enumerate(kinds) if k is ModificationTrack)
    assert diff < min(i for i, k in enumerate(kinds) if k is CoverageTrack)


def test_a_dmr_track_needs_no_pileup(view_kwargs, mod_diff_table):
    from isotracks.tracks import ModDiffTrack, ModificationTrack

    canvas = IsoTracks(**{**view_kwargs,
                          "mod_diff": {"Ctrl vs Trt": mod_diff_table}}).build()
    assert len(canvas.of_type(ModDiffTrack)) == 1
    assert canvas.of_type(ModificationTrack) == []


def test_the_dmr_axis_is_fixed_and_signed(view_kwargs, mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot(region=("chrT", 6_000, 6_400, "+"))
    track = view.canvas.of_type(ModDiffTrack)[0]
    assert track.data_ylim == (-1.0, 1.0)


def test_loss_and_gain_are_drawn_on_opposite_sides(view_kwargs, mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot(region=("chrT", 6_000, 6_400, "+"))
    sites = view.canvas.of_type(ModDiffTrack)[0].region_data
    assert (sites.lost["effect_size"] > 0).all()
    assert (sites.gained["effect_size"] < 0).all()


def test_the_dmr_gutter_reports_sites_drawn_against_sites_tested(view_kwargs,
                                                                 mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot(region=("chrT", 6_000, 6_400, "+"))
    label = view.canvas.of_type(ModDiffTrack)[0].config["sublabel"]
    assert label == "2/4 sites"


def test_the_dmr_label_is_the_comparison(view_kwargs, mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    canvas = IsoTracks(**{**view_kwargs,
                          "mod_diff": {"Control vs Treated": mod_diff_table}}).build()
    assert canvas.of_type(ModDiffTrack)[0].label == "Control vs Treated"


def test_loss_and_gain_colours_separate(view_kwargs):
    from isotracks import Theme
    from .test_theme import _delta_e

    theme = Theme()
    assert _delta_e(theme.mod_loss_color, theme.mod_gain_color) > 40


def test_the_dmr_key_puts_each_direction_on_its_own_side(view_kwargs,
                                                         mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot(region=("chrT", 6_000, 6_400, "+"))

    track = view.canvas.of_type(ModDiffTrack)[0]
    placed = {text.get_text(): text.get_position()[1]
              for text in track.ax.texts}
    assert placed["lost"] == 1.0  # above zero, where the loss stems go
    assert placed["gained"] == 0.0  # below it, with the gain stems


def test_in_axes_keys_are_haloed_so_stems_can_pass_under_them(view_kwargs,
                                                              mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot(region=("chrT", 6_000, 6_400, "+"))

    track = view.canvas.of_type(ModDiffTrack)[0]
    assert track.ax.texts, "the key drew nothing"
    assert all(text.get_path_effects() for text in track.ax.texts)


def test_a_transparent_theme_gets_no_white_halo(view_kwargs, mod_diff_table):
    from isotracks import Theme, theme as theme_module
    from isotracks.tracks import ModDiffTrack

    previous = theme_module.active()
    theme_module.use(Theme(transparent=True))
    try:
        view = IsoTracks(**{**view_kwargs,
                            "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
        view.plot(region=("chrT", 6_000, 6_400, "+"))
        track = view.canvas.of_type(ModDiffTrack)[0]
        assert not any(text.get_path_effects() for text in track.ax.texts)
    finally:
        theme_module.use(previous)


# --------------------------------------- what the per-position tracks cover

def test_merge_intervals_unions_overlapping_and_adjacent_spans():
    from isotracks.io._intervals import merge_intervals

    assert merge_intervals([(10, 20), (15, 25), (40, 50), (21, 30)]) == [
        (10, 30), (40, 50),
    ]


def test_an_interval_mask_with_no_spans_restricts_nothing():
    from isotracks.io._intervals import IntervalMask

    mask = IntervalMask([])
    assert not mask
    assert mask.contains(12345)


def test_an_interval_mask_excludes_the_gaps():
    from isotracks.io._intervals import IntervalMask

    mask = IntervalMask([(100, 200), (300, 400)])
    assert mask.contains(100) and mask.contains(200) and mask.contains(350)
    assert not mask.contains(99) and not mask.contains(250) and not mask.contains(401)


def test_drawn_exons_covers_only_the_isoforms_drawn(view_kwargs):
    view = IsoTracks(**view_kwargs)
    every = view.drawn_exons(view.transcripts())
    just_one = view.drawn_exons(["TX_SHORT"])
    assert just_one
    assert sum(e - s for s, e in just_one) < sum(e - s for s, e in every)


def test_modifications_are_kept_inside_the_drawn_exons(view_kwargs, modkit_pileups):
    from isotracks.tracks import ModificationTrack

    inside = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    inside.plot()
    exonic = inside.canvas.of_type(ModificationTrack)[0].region_data

    everywhere = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                              "mod_regions": "window"})
    everywhere.plot()
    windowed = everywhere.canvas.of_type(ModificationTrack)[0].region_data

    assert len(exonic.frame) < len(windowed.frame)
    mask = _exon_mask(inside.drawn_exons())
    assert all(mask(p) for p in exonic.frame["position"])
    assert not all(mask(p) for p in windowed.frame["position"])


def test_restricting_to_exons_also_restricts_the_assessed_count(view_kwargs,
                                                                modkit_pileups):
    from isotracks.tracks import ModificationTrack

    inside = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups})
    inside.plot()
    everywhere = IsoTracks(**{**view_kwargs, "modkit_pileup": modkit_pileups,
                              "mod_regions": "window"})
    everywhere.plot()

    assert (inside.canvas.of_type(ModificationTrack)[0].region_data.assessed
            < everywhere.canvas.of_type(ModificationTrack)[0].region_data.assessed)


def test_differential_sites_are_kept_inside_the_drawn_exons(view_kwargs,
                                                            mod_diff_table):
    from isotracks.tracks import ModDiffTrack

    view = IsoTracks(**{**view_kwargs, "mod_diff": {"Ctrl vs Trt": mod_diff_table}})
    view.plot()
    sites = view.canvas.of_type(ModDiffTrack)[0].region_data
    mask = _exon_mask(view.drawn_exons())
    assert all(mask(p) for p in sites.frame["position"])


def _exon_mask(spans):
    return lambda position: any(s <= position <= e for s, e in spans)


# --------------------------------------- which reads the sashimi comes from

def test_coverage_uses_only_the_drawn_isoforms_reads(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1})
    drawn = view.transcripts()
    assert len(drawn) == 1

    names = view._coverage_read_names(drawn)
    assert names == set(view.gene_reads.reads_by_transcript[drawn[0]])
    assert len(names) < len(view.gene_reads.read_names)


def test_coverage_reads_gene_restores_every_read_of_the_gene(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1,
                        "coverage_reads": "gene"})
    names = view._coverage_read_names(view.transcripts())
    assert names == set(view.gene_reads.read_names)


def test_unknown_option_values_are_refused_at_construction(view_kwargs):
    with pytest.raises(ValueError, match="unknown mod_regions"):
        IsoTracks(**{**view_kwargs, "mod_regions": "introns"})
    with pytest.raises(ValueError, match="unknown coverage_reads"):
        IsoTracks(**{**view_kwargs, "coverage_reads": "everything"})


# --------------------------------------- the gene model row

def _isoform_models(canvas):
    from isotracks.tracks import GeneModelTrack

    return [t for t in canvas.of_type(TranscriptTrack)
            if not isinstance(t, GeneModelTrack)]


def test_the_gene_model_is_labelled_with_the_gene_name(view_kwargs):
    from isotracks.tracks import GeneModelTrack
    from .conftest import GENE

    canvas = IsoTracks(**view_kwargs).build()
    track = canvas.of_type(GeneModelTrack)[0]
    assert track.label == GENE
    assert track.config["label_weight"] == "bold"


def test_the_gene_model_sits_directly_above_the_isoform_models(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**view_kwargs).build()
    order = canvas.tracks.index
    gene = order(canvas.of_type(GeneModelTrack)[0])
    assert gene < min(order(t) for t in _isoform_models(canvas))


def test_the_gene_model_overlays_only_the_isoforms_drawn(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1})
    view.plot()
    track = view.canvas.of_type(GeneModelTrack)[0]

    drawn = [t.label for t in _isoform_models(view.canvas)]
    assert len(drawn) == 1
    assert list(track.region_data.df["id"]) == view.transcripts()
    rows = set(zip(track.region_data.df["y_min"], track.region_data.df["y_max"]))
    assert len(rows) == 1, "the models must overlap, not stack"


def test_the_gene_model_carries_no_isoform_count(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**{**view_kwargs, "max_isoforms": 1}).build()
    assert canvas.of_type(GeneModelTrack)[0].config["sublabel"] == ""


def test_with_nothing_drawn_the_gene_model_falls_back_to_the_annotation(view_kwargs):
    from isotracks.tracks import GeneModelTrack
    from .conftest import ISOFORMS

    view = IsoTracks(**{**view_kwargs, "read_annot": None})
    assert view.transcripts() == []
    view.plot()
    track = view.canvas.of_type(GeneModelTrack)[0]
    assert set(track.region_data.df["id"]) == set(ISOFORMS)


def test_the_gene_model_says_nothing_when_it_adds_nothing(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**view_kwargs).build()
    assert canvas.of_type(GeneModelTrack)[0].config["sublabel"] == ""


def test_the_gene_model_is_translucent_so_overlap_reads_as_agreement(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**view_kwargs).build()
    track = canvas.of_type(GeneModelTrack)[0]
    assert track.config["style"]["exon"]["facecolor"][3] < 1.0


def test_the_gene_model_can_be_turned_off(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    canvas = IsoTracks(**{**view_kwargs, "gene_model": False}).build()
    assert canvas.of_type(GeneModelTrack) == []
    assert len(_isoform_models(canvas)) == 2


def test_chevrons_are_on_the_gene_row_and_not_on_the_isoform_models(view_kwargs):
    from isotracks.tracks import GeneModelTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    gene = view.canvas.of_type(GeneModelTrack)[0]
    assert gene.config["direction_arrows"] is True
    assert gene.ax.lines, "the gene row should carry direction chevrons"

    for model in _isoform_models(view.canvas):
        assert model.config["direction_arrows"] is False
        assert not model.ax.lines


def test_the_axis_line_names_the_gene_as_well_as_the_locus(view_kwargs):
    from isotracks.tracks import CoordinateAxisTrack
    from .conftest import CHROM, GENE

    view = IsoTracks(**{**view_kwargs, "gene_model": False})
    view.plot()
    caption = view.canvas.of_type(CoordinateAxisTrack)[0]._caption()
    assert GENE in caption and CHROM in caption


# --------------------------------------- compact layout and label headroom

def test_compact_shortens_the_shape_tracks_but_not_the_type():
    from isotracks.layout import Layout

    normal, compact = Layout(), Layout.compact()
    assert compact.read_stack < normal.read_stack
    assert compact.coverage < normal.coverage
    assert compact.modification < normal.modification
    assert compact.tss_sites == normal.tss_sites
    assert compact.axis == normal.axis


def test_compact_overrides_can_still_be_given():
    from isotracks.layout import Layout

    assert Layout.compact(read_stack=2.5).read_stack == 2.5


def test_compact_makes_a_shorter_figure(view_kwargs, tmp_path):
    tall = IsoTracks(**{**view_kwargs, "compact": False}).plot(
        output=str(tmp_path / "tall.pdf"))
    short = IsoTracks(**view_kwargs).plot(output=str(tmp_path / "short.pdf"))
    assert short.get_size_inches()[1] < tall.get_size_inches()[1]
    # The width and the type are unchanged.
    assert short.get_size_inches()[0] == pytest.approx(tall.get_size_inches()[0])


def test_an_explicit_layout_beats_compact(view_kwargs):
    from isotracks.layout import Layout

    view = IsoTracks(**{**view_kwargs, "compact": True, "layout": Layout()})
    assert view.layout.read_stack == Layout().read_stack


def test_a_two_line_gutter_grows_with_the_readability_tier():
    from isotracks import Theme, theme as theme_module
    from isotracks.layout import label_block_units

    previous = theme_module.active()
    try:
        theme_module.use(Theme(readability="strict"))
        strict = label_block_units(2)
        theme_module.use(Theme(readability="large"))
        large = label_block_units(2)
    finally:
        theme_module.use(previous)
    assert large > strict
    assert label_block_units(1) == 0.0


def test_a_track_is_never_shorter_than_its_gutter_label_needs(view_kwargs):
    from isotracks import Theme, theme as theme_module
    from isotracks.layout import label_block_units
    from isotracks.tracks import ReadTrack

    previous = theme_module.active()
    try:
        theme_module.use(Theme(readability="large"))
        view = IsoTracks(**{**view_kwargs, "compact": True})
        view.plot()
        needed = label_block_units(2)
        for track in view.canvas.of_type(ReadTrack):
            assert track.config["label_lines"] == 2
            span = abs(track.ax.get_position().height) * view.canvas.fig.get_figheight()
            assert span >= needed * 4.2 / 25.4 * 0.9  # units -> mm -> inches
    finally:
        theme_module.use(previous)


def test_an_ordinary_stack_is_piled_in_full(view_kwargs):
    from isotracks.browser import DEFAULT_MAX_READS
    from isotracks.tracks import ReadTrack

    view = IsoTracks(**view_kwargs)
    view.plot()
    for track in view.canvas.of_type(ReadTrack):
        assert track._available <= DEFAULT_MAX_READS
        assert track.read_count == track._available  # nothing was dropped
        assert "/" not in track.config["sublabel"]  # so nothing is reported


def test_a_thinned_stack_keeps_the_requested_number_of_reads(view_kwargs):
    from isotracks.tracks import ReadTrack

    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=5)
    track = view.canvas.of_type(ReadTrack)[0]
    assert track.read_count == 5 < track._available
    assert track.config["sublabel"] == f"n = 5/{track._available:,}"


def test_max_reads_none_draws_every_read(view_kwargs):
    from isotracks.tracks import ReadTrack

    view = IsoTracks(**view_kwargs)
    view.plot(max_reads=None)
    for track in view.canvas.of_type(ReadTrack):
        assert track.read_count == track._available


# --------------------------------------- gene, isoform and exon information

def test_compact_is_the_default(view_kwargs):
    from isotracks.layout import Layout

    assert IsoTracks(**view_kwargs).layout.read_stack == Layout.compact().read_stack


def test_exon_lengths_are_inclusive_of_both_ends(view_kwargs):
    from .conftest import ISOFORMS

    table = IsoTracks(**view_kwargs).exons()
    for row in table.itertuples(index=False):
        assert row.length == row.end - row.start + 1
    longest = max(e - s + 1 for spans in ISOFORMS.values() for s, e in spans)
    assert table["length"].max() == longest


def test_a_transcript_length_is_its_exons_summed_not_its_span(view_kwargs):
    from .conftest import ISOFORMS

    table = IsoTracks(**view_kwargs).isoform_table().set_index("transcript_id")
    for tid, spans in ISOFORMS.items():
        row = table.loc[tid]
        assert row["length"] == sum(e - s + 1 for s, e in spans)
        assert row["genomic_span"] == spans[-1][1] - spans[0][0] + 1
        assert row["exons"] == len(spans)
    # A spliced transcript is shorter than the stretch of genome it covers.
    assert (table["length"] < table["genomic_span"]).all()


def test_the_isoform_table_reports_read_support(view_kwargs):
    view = IsoTracks(**view_kwargs)
    table = view.isoform_table().set_index("transcript_id")
    assert table["reads"].sum() == len(view.gene_reads.read_names)
    assert table["drawn"].all()  # both pass the threshold here


def test_the_isoform_table_flags_what_the_figure_leaves_out(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "max_isoforms": 1})
    table = view.isoform_table(all_isoforms=True)
    assert set(table["drawn"]) == {True, False}
    assert table.loc[table["drawn"], "reads"].iloc[0] >= table["reads"].min()


def test_exons_can_be_asked_for_one_isoform(view_kwargs):
    table = IsoTracks(**view_kwargs).exons("TX_SHORT")
    assert set(table["transcript_id"]) == {"TX_SHORT"}


def test_the_report_names_the_gene_and_every_exon(view_kwargs):
    from .conftest import GENE, ISOFORMS

    text = IsoTracks(**view_kwargs).describe()
    assert GENE in text
    for tid in ISOFORMS:
        assert tid in text
    # Every exon of every isoform gets a line.
    assert text.count("exon ") == sum(len(s) for s in ISOFORMS.values())


def test_the_report_can_leave_the_exon_listing_out(view_kwargs):
    text = IsoTracks(**view_kwargs).describe(exons=False)
    assert "exon " not in text
    assert "TX_LONG" in text


def test_an_empty_gene_reports_rather_than_raising(view_kwargs, gtf_path):
    view = IsoTracks(**{**view_kwargs, "gene_name": "TESTG",
                        "region": ("chrT", 1_000, 2_000, "+")})
    view.gene_name = "NOT_A_GENE"
    view._transcripts = None
    assert view.isoform_table().empty


# ------------------------------- type staying inside the track that holds it

def _text_box(figure, artist):
    return artist.get_window_extent(renderer=figure.canvas.get_renderer())


def test_the_isoform_heading_clears_the_rule_above_it(view_kwargs):
    from isotracks.tracks import SpacerTrack

    view = IsoTracks(**view_kwargs)
    figure = view.plot()
    figure.canvas.draw()

    headed = [t for t in view.canvas.of_type(SpacerTrack) if t.config["heading"]]
    ruled = [t for t in headed if t.config["rule_above"]]
    assert ruled, "no group had a rule above it to test against"

    for track in ruled:
        rule_y = track.ax.transData.transform((0, track.ylim[1]))[1]
        top = max(_text_box(figure, text).y1 for text in track.ax.texts)
        assert top <= rule_y, (
            f"heading {track.config['heading']!r} rises {top - rule_y:.1f} px "
            "above the rule"
        )


def test_the_heading_is_smaller_than_the_gutter_labels_beside_it(view_kwargs):
    from isotracks.tracks import SpacerTrack

    canvas = IsoTracks(**view_kwargs).build()
    track = [t for t in canvas.of_type(SpacerTrack) if t.config["heading"]][0]
    assert track.heading_size() < track.theme.fonts["annot"]


def test_the_scale_bar_label_stays_inside_its_own_track(view_kwargs):
    from isotracks.tracks import ScaleBarTrack

    view = IsoTracks(**view_kwargs)
    figure = view.plot()
    figure.canvas.draw()

    track = view.canvas.of_type(ScaleBarTrack)[0]
    axes_box = track.ax.get_window_extent()
    for text in track.ax.texts:
        box = _text_box(figure, text)
        assert box.y1 <= axes_box.y1 + 1.0, "the scale label rises out of its row"
        assert box.y0 >= axes_box.y0 - 1.0, "the scale label drops out of its row"


def test_a_track_is_never_shorter_than_the_type_it_holds(view_kwargs):
    from isotracks.tracks import ScaleBarTrack, SpacerTrack

    canvas = IsoTracks(**view_kwargs).build()
    for track in canvas.of_type(ScaleBarTrack) + canvas.of_type(SpacerTrack):
        floor = track.min_height()
        if floor:
            assert max(track.config["height"], floor) >= floor
    # The scale bar's own floor is real, not zero.
    assert canvas.of_type(ScaleBarTrack)[0].min_height() > 0
