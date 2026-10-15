"""Argument parsing and the command-line entry points."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from isotracks import cache as cache_module
from isotracks.annotation import ReadAnnotation
from isotracks.cli import (
    main,
    parse_bam_spec,
    parse_mapping,
    parse_region,
    read_gene_list,
)

from .conftest import GENE


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_bam_spec_splits_conditions_and_replicates():
    assert parse_bam_spec("Ctrl:a.bam,b.bam;Trt:c.bam") == {
        "Ctrl": ["a.bam", "b.bam"], "Trt": ["c.bam"],
    }


def test_bam_spec_merges_a_repeated_condition():
    assert parse_bam_spec("Ctrl:a.bam;Ctrl:b.bam") == {"Ctrl": ["a.bam", "b.bam"]}


@pytest.mark.parametrize("bad", ["a.bam", "Ctrl:", ":a.bam", ""])
def test_bam_spec_rejects_malformed_input(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bam_spec(bad)


def test_region_accepts_thousands_separators_and_a_strand():
    assert parse_region("chr1:1,000,000-1,050,000:-") == ("chr1", 1_000_000, 1_050_000, "-")


def test_a_region_without_a_strand_is_left_unknown_not_assumed_plus():
    assert parse_region("chr1:100-200")[3] is None


@pytest.mark.parametrize("bad", ["chr1", "chr1:100", "chr1:a-b", "chr1:1-2:?"])
def test_region_rejects_malformed_input(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_region(bad)


def test_mapping_parses_key_value_pairs():
    assert parse_mapping("Ctrl:Control,Trt:Treated") == {"Ctrl": "Control", "Trt": "Treated"}


def test_mapping_of_nothing_is_none():
    assert parse_mapping(None) is None
    assert parse_mapping("") is None


def test_gene_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "genes.txt"
    path.write_text("SAT1\n\n# a comment\nMYC\n  \nTP53\n")
    assert read_gene_list(str(path)) == ["SAT1", "MYC", "TP53"]


def _cli_inputs(view_kwargs):
    bam = ";".join(
        f"{cond}:{','.join(paths)}" for cond, paths in view_kwargs["bam_files"].items()
    )
    return [
        "--bam", bam,
        "--annotation-gtf", view_kwargs["annotation_gtf"],
        "--read-annot", view_kwargs["read_annot"],
        "--bed", view_kwargs["bed_file"],
        "--polya", view_kwargs["polya_clusters"],
        "--tss", view_kwargs["tss_clusters"],
        "--min-reads", "1",
    ]


def test_plot_command_writes_a_figure(view_kwargs, tmp_path):
    output = tmp_path / "out.png"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--output", str(output), "--quiet"]) == 0
    assert output.exists()


def test_plot_command_honours_zoom(view_kwargs, tmp_path):
    output = tmp_path / "zoom.pdf"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--zoom", "last_exon", "--output", str(output), "--quiet"]) == 0
    assert output.exists()


def test_plot_command_writes_several_formats(view_kwargs, tmp_path):
    stem = tmp_path / "both"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--output", str(stem), "--formats", "pdf", "png", "--quiet"]) == 0
    assert (tmp_path / "both.pdf").exists()
    assert (tmp_path / "both.png").exists()


def test_a_stem_with_no_extension_writes_pdf_and_png(view_kwargs, tmp_path):
    stem = tmp_path / "gene"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--output", str(stem), "--quiet"]) == 0
    assert (tmp_path / "gene.pdf").exists()
    assert (tmp_path / "gene.png").exists()


def test_plot_command_accepts_a_polya_scale(view_kwargs, tmp_path):
    output = tmp_path / "scaled.png"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--polya-scale", "5", "--output", str(output), "--quiet"]) == 0
    assert output.exists()


def test_batch_command_writes_one_figure_per_gene(view_kwargs, tmp_path):
    genes = tmp_path / "genes.txt"
    genes.write_text(f"{GENE}\n")
    outdir = tmp_path / "figures"
    assert main(["batch", "--genes", str(genes), *_cli_inputs(view_kwargs),
                 "--outdir", str(outdir), "--format", "png", "--quiet"]) == 0
    assert (outdir / f"{GENE}.png").exists()


def test_batch_keeps_going_and_reports_a_failed_gene(view_kwargs, tmp_path, capsys):
    genes = tmp_path / "genes.txt"
    genes.write_text(f"NO_SUCH_GENE\n{GENE}\n")
    outdir = tmp_path / "figures"
    exit_code = main(["batch", "--genes", str(genes), *_cli_inputs(view_kwargs),
                      "--outdir", str(outdir), "--format", "png", "--quiet"])
    assert exit_code == 1  # a failure is reported
    assert (outdir / f"{GENE}.png").exists()  # the good gene still ran
    assert "NO_SUCH_GENE" in capsys.readouterr().err


def test_batch_skips_existing_figures(view_kwargs, tmp_path, capsys):
    genes = tmp_path / "genes.txt"
    genes.write_text(f"{GENE}\n")
    outdir = tmp_path / "figures"
    outdir.mkdir()
    (outdir / f"{GENE}.png").write_bytes(b"already here")

    main(["batch", "--genes", str(genes), *_cli_inputs(view_kwargs),
          "--outdir", str(outdir), "--format", "png", "--skip-existing", "--quiet"])
    assert (outdir / f"{GENE}.png").read_bytes() == b"already here"
    assert "skipping" in capsys.readouterr().out


def test_polya_command_writes_a_csv(view_kwargs, tmp_path):
    output = tmp_path / "polya.csv"
    assert main(["polya", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--output", str(output), "--quiet"]) == 0
    table = pd.read_csv(output)
    assert set(table.columns) >= {"gene", "transcript_id", "condition", "polya_length"}


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "COMMAND" in capsys.readouterr().out


# --------------------------------------------------------------------- index

def test_index_builds_and_is_then_used(read_annot_path, tmp_path, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv(cache_module.CACHE_DIR_ENV, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    assert main(["index", "--read-annot", read_annot_path]) == 0
    manifest = cache_module.find_cache(read_annot_path)
    assert manifest is not None and manifest.rows == 48

    # The cached read must agree with the scan it replaces.
    cached = ReadAnnotation(read_annot_path).for_gene(GENE)
    scanned = ReadAnnotation(read_annot_path, use_cache=False).for_gene(GENE)
    assert cached.counts() == scanned.counts()


def test_index_is_not_rebuilt_when_current(read_annot_path, tmp_path, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv(cache_module.CACHE_DIR_ENV, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    main(["index", "--read-annot", read_annot_path])
    capsys.readouterr()
    main(["index", "--read-annot", read_annot_path])
    assert "already current" in capsys.readouterr().out


def test_index_drop_removes_it(read_annot_path, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv(cache_module.CACHE_DIR_ENV, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    main(["index", "--read-annot", read_annot_path])
    assert main(["index", "--read-annot", read_annot_path, "--drop"]) == 0
    assert cache_module.find_cache(read_annot_path) is None


def test_a_stale_index_is_ignored(read_annot_path, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv(cache_module.CACHE_DIR_ENV, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    main(["index", "--read-annot", read_annot_path])
    assert cache_module.find_cache(read_annot_path) is not None

    with open(read_annot_path, "a") as handle:
        handle.write("extra\tCtrl_rep1\tchrT\t+\tTESTG\tTX_LONG\tx\n")
    assert cache_module.find_cache(read_annot_path) is None


def test_bucketing_is_stable_across_processes():
    assert cache_module.bucket_of("SAT1", 64) == cache_module.bucket_of("SAT1", 64)
    assert 0 <= cache_module.bucket_of("SAT1", 64) < 64


# --------------------------------------------------------------- modifications

def test_mod_groups_share_a_track_when_comma_joined():
    from isotracks.cli import parse_mod_groups

    assert parse_mod_groups(["a,17802"]) == [["a", "17802"]]


def test_separate_arguments_get_a_track_each():
    from isotracks.cli import parse_mod_groups

    assert parse_mod_groups(["a", "17802"]) == [["a"], ["17802"]]


def test_groups_can_mix_both_forms():
    from isotracks.cli import parse_mod_groups

    assert parse_mod_groups(["a,17802", "m"]) == [["a", "17802"], ["m"]]


def test_no_mods_argument_leaves_the_default_alone():
    from isotracks.cli import parse_mod_groups

    assert parse_mod_groups(None) is None
    assert parse_mod_groups([]) is None


def test_an_empty_mod_group_is_refused():
    from isotracks.cli import parse_mod_groups

    with pytest.raises(argparse.ArgumentTypeError, match="no modification codes"):
        parse_mod_groups([","])


def test_the_cli_draws_modifications_when_given_a_pileup(view_kwargs, modkit_pileups, tmp_path):
    spec = ";".join(f"{c}:{','.join(p)}" for c, p in modkit_pileups.items())
    output = tmp_path / "mods.png"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--modkit", spec, "--mods", "a", "17802",
                 "--output", str(output), "--quiet"]) == 0
    assert output.exists()


# --------------------------------------------------------- the info command

def test_info_prints_a_report(view_kwargs, capsys):
    from .conftest import GENE, ISOFORMS

    assert main(["info", "--gene", GENE, *_cli_inputs(view_kwargs), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert GENE in out
    for tid in ISOFORMS:
        assert tid in out


def test_info_needs_no_bam_to_report_coordinates(view_kwargs, capsys):
    from .conftest import GENE

    assert main(["info", "--gene", GENE,
                 "--annotation-gtf", view_kwargs["annotation_gtf"],
                 "--read-annot", view_kwargs["read_annot"], "--quiet"]) == 0
    assert GENE in capsys.readouterr().out


def test_info_writes_an_exon_table(view_kwargs, tmp_path):
    import pandas as pd
    from .conftest import GENE, ISOFORMS

    output = tmp_path / "exons.tsv"
    assert main(["info", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--format", "tsv", "--exons-table",
                 "--output", str(output), "--quiet"]) == 0

    table = pd.read_csv(output, sep="\t")
    assert len(table) == sum(len(s) for s in ISOFORMS.values())
    assert (table["length"] == table["end"] - table["start"] + 1).all()


def test_info_writes_an_isoform_table(view_kwargs, tmp_path):
    import pandas as pd
    from .conftest import GENE, ISOFORMS

    output = tmp_path / "isoforms.csv"
    assert main(["info", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--format", "csv", "--output", str(output), "--quiet"]) == 0

    table = pd.read_csv(output)
    assert len(table) == len(ISOFORMS)
    assert (table["length"] < table["genomic_span"]).all()


def test_info_can_report_one_transcript(view_kwargs, capsys):
    from .conftest import GENE

    assert main(["info", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--transcript", "TX_SHORT", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "TX_SHORT" in out and "TX_LONG" not in out


# ------------------------------------------------------------ highlights

def test_plot_marks_several_regions(view_kwargs, tmp_path, monkeypatch):
    from isotracks import IsoTracks

    seen = {}
    real_plot = IsoTracks.plot

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(IsoTracks, "plot", spy)
    output = tmp_path / "marked.png"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
                 "--highlight", "3_utr", "exon_2:TX_SHORT",
                 "--highlight-style", "outline",
                 "--output", str(output), "--quiet"]) == 0
    assert output.exists()
    assert seen["highlight"] == ["3_utr", "exon_2:TX_SHORT"]
    assert seen["highlight_style"] == "outline"


def test_a_bad_highlight_name_is_reported_not_drawn(view_kwargs, tmp_path):
    output = tmp_path / "never.png"
    with pytest.raises(SystemExit, match="unknown highlight"):
        main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs),
              "--highlight", "promoter", "--output", str(output), "--quiet"])
    assert not output.exists()


# ---------------------------------------------------------------- fusion

def test_plot_draws_a_fusion_by_name(fusion_kwargs, tmp_path):
    bam = ";".join(f"{c}:{','.join(p)}" for c, p in fusion_kwargs["bam_files"].items())
    output = tmp_path / "fusion.png"
    assert main(["plot", "--fusion", fusion_kwargs["fusion"],
                 "--fusion-table", fusion_kwargs["fusion_table"],
                 "--bam", bam, "--reference-gtf", fusion_kwargs["reference_gtf"],
                 "--output", str(output), "--quiet"]) == 0
    assert output.exists() and output.stat().st_size > 1_000


def test_a_fusion_needs_its_table(fusion_kwargs, tmp_path):
    bam = ";".join(f"{c}:{','.join(p)}" for c, p in fusion_kwargs["bam_files"].items())
    with pytest.raises(SystemExit, match="--fusion-table"):
        main(["plot", "--fusion", fusion_kwargs["fusion"], "--bam", bam,
              "--reference-gtf", fusion_kwargs["reference_gtf"],
              "--output", str(tmp_path / "x.png"), "--quiet"])


@pytest.fixture
def drawn_fusion(monkeypatch):
    """The FusionView the CLI drew, once ``main`` returns."""
    from isotracks import fusion_view

    seen = {}
    real_plot = fusion_view.FusionView.plot

    def spy(self, *args, **kwargs):
        seen["view"] = self
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(fusion_view.FusionView, "plot", spy)
    return seen


def test_plot_draws_a_fusion_from_its_tables_without_bams(fusion_tables, fusion_kwargs,
                                                         tmp_path, drawn_fusion):
    output = tmp_path / "fusion.png"
    assert main(["plot", "--fusion", "M000001", "--fusion-table", fusion_tables,
                 "--conditions", "Ctrl:fusion_Ctrl;Trt:fusion_Trt",
                 "--reference-gtf", fusion_kwargs["reference_gtf"],
                 "--output", str(output), "--quiet"]) == 0
    assert output.exists() and output.stat().st_size > 1_000
    view = drawn_fusion["view"]
    assert view.condition_order == ["Ctrl", "Trt"]
    assert [g.name for g in view.groups] == ["GENE5::GENEM::GENE3", "GENEM::GENE3"]


def test_fusion_reads_complete_keeps_only_the_whole_fusion(fusion_tables, tmp_path,
                                                         drawn_fusion):
    assert main(["plot", "--fusion", "GENE5::GENEM::GENE3", "--fusion-table", fusion_tables,
                 "--fusion-reads", "complete",
                 "--output", str(tmp_path / "fusion.png"), "--quiet"]) == 0
    assert [g.name for g in drawn_fusion["view"].groups] == ["GENE5::GENEM::GENE3"]


def test_a_gene_still_needs_bams(tmp_path):
    with pytest.raises(SystemExit, match="--bam"):
        main(["plot", "--gene", "SAT1", "--output", str(tmp_path / "x.png"), "--quiet"])


def test_dataset_spec_groups_datasets_into_conditions():
    from isotracks.cli import parse_dataset_spec

    assert parse_dataset_spec("Control:a,b; Treated:c") == {"Control": ["a", "b"], "Treated": ["c"]}
    with pytest.raises(argparse.ArgumentTypeError, match="datasets"):
        parse_dataset_spec("Control:")


def test_the_cli_draws_without_a_gui_backend(monkeypatch):
    import matplotlib

    monkeypatch.delenv("MPLBACKEND", raising=False)
    matplotlib.use("svg")
    main([])
    assert matplotlib.get_backend().lower() == "agg"


def test_an_explicit_mplbackend_is_respected(monkeypatch):
    import matplotlib

    monkeypatch.setenv("MPLBACKEND", "svg")
    matplotlib.use("svg")
    try:
        main([])
        assert matplotlib.get_backend().lower() == "svg"
    finally:
        matplotlib.use("Agg")


def test_the_fusion_schematic_can_be_left_out_from_the_cli(fusion_tables, tmp_path,
                                                           drawn_fusion):
    assert main(["plot", "--fusion", "M000001", "--fusion-table", fusion_tables,
                 "--no-fusion-schematic",
                 "--output", str(tmp_path / "fusion.png"), "--quiet"]) == 0
    assert drawn_fusion["view"].schematic is False


def test_a_fusion_takes_the_site_beds_from_the_cli(fusion_tables, tmp_path, drawn_fusion):
    bed = tmp_path / "sites.bed"
    bed.write_text("chrB\t4595\t4605\t4600\t50\t+\texon\tG\t2.1\t50\t1.0\t3\t.\n")
    assert main(["plot", "--fusion", "M000001", "--fusion-table", fusion_tables,
                 "--tss", str(bed), "--polya", str(bed),
                 "--output", str(tmp_path / "fusion.png"), "--quiet"]) == 0
    view = drawn_fusion["view"]
    assert view.tss_clusters == str(bed) and view.polya_clusters == str(bed)



def test_input_mistakes_are_reported_without_a_traceback(tmp_path):
    with pytest.raises(SystemExit, match="isotracks: no IsoForge output"):
        main(["plot", "--gene", "SAT1", "--isoforge", str(tmp_path / "missing"),
              "--output", str(tmp_path / "x.png")])


def test_width_is_a_preset_or_millimetres():
    from isotracks.cli import parse_width

    assert parse_width("single") == "single"
    assert parse_width("176") == 176.0
    assert parse_width("120.5") == 120.5
    for bad in ("wide", "0", "-40"):
        with pytest.raises(argparse.ArgumentTypeError, match="not a width"):
            parse_width(bad)


@pytest.mark.parametrize("width, millimetres", [("176", 176.0), ("double", 183.0)])
def test_plot_draws_at_the_width_asked_for(view_kwargs, tmp_path, width, millimetres):
    from isotracks import theme

    output = tmp_path / "out.png"
    assert main(["plot", "--gene", GENE, *_cli_inputs(view_kwargs), "--width", width,
                 "--output", str(output), "--quiet"]) == 0
    assert theme.active().width_mm == millimetres
    figure = plt.gcf()
    assert figure.get_figwidth() * 25.4 == pytest.approx(millimetres)


def test_batch_takes_a_width_in_millimetres(view_kwargs, tmp_path):
    genes = tmp_path / "genes.txt"
    genes.write_text(GENE + "\n")
    assert main(["batch", "--genes", str(genes), "--outdir", str(tmp_path / "figures"),
                 "--format", "png", *_cli_inputs(view_kwargs), "--width", "120", "--quiet"]) == 0
    assert (tmp_path / "figures" / f"{GENE}.png").stat().st_size > 1_000
