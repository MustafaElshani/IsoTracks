"""An IsoForge run located from its prefix."""

from __future__ import annotations

import os
import shutil

import pytest

from isotracks import FusionView, IsoTracks
from isotracks.cli import main
from isotracks.isoforge import find_outputs, run_prefix
from isotracks.tracks import TranscriptTrack

from .conftest import GENE


@pytest.fixture
def isoforge_run(tmp_path, gtf_path, isoforge_parquet, polya_bed_path, tss_bed_path):
    """A run laid out as ``isoforge annotate --o <dir>/RUN`` writes it."""
    prefix = str(tmp_path / "RUN")
    shutil.copy(gtf_path, prefix + "_isoforge.gtf")
    shutil.copy(isoforge_parquet["parquet"], prefix + "_isoforge_read_annot.parquet")
    os.makedirs(prefix + "_TSS_TES_Clusters")
    shutil.copy(polya_bed_path, prefix + "_TSS_TES_Clusters/tes_clusters.bed")
    shutil.copy(tss_bed_path, prefix + "_TSS_TES_Clusters/tss_clusters.bed")
    return prefix


def test_outputs_are_found_from_the_prefix_the_directory_or_one_file(isoforge_run):
    expected = {
        "annotation_gtf": isoforge_run + "_isoforge.gtf",
        "read_annot": isoforge_run + "_isoforge_read_annot.parquet",
        "polya_clusters": isoforge_run + "_TSS_TES_Clusters/tes_clusters.bed",
        "tss_clusters": isoforge_run + "_TSS_TES_Clusters/tss_clusters.bed",
    }
    for source in (isoforge_run, os.path.dirname(isoforge_run), isoforge_run + "_isoforge.gtf"):
        assert find_outputs(source) == expected


def test_a_tsv_read_annotation_is_used_when_there_is_no_parquet(isoforge_run):
    os.rename(isoforge_run + "_isoforge_read_annot.parquet",
              isoforge_run + "_isoforge_read_annot.tsv")
    assert find_outputs(isoforge_run)["read_annot"].endswith("_read_annot.tsv")


def test_a_path_with_no_run_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no IsoForge output"):
        find_outputs(str(tmp_path / "missing"))


def test_a_directory_with_two_runs_needs_a_prefix(isoforge_run):
    shutil.copy(isoforge_run + "_isoforge.gtf",
                os.path.join(os.path.dirname(isoforge_run), "OTHER_isoforge.gtf"))
    with pytest.raises(ValueError, match="one IsoForge run"):
        run_prefix(os.path.dirname(isoforge_run))


def test_a_gene_is_drawn_from_the_prefix_and_bams(isoforge_run, bam_files):
    view = IsoTracks(gene_name=GENE, bam_files=bam_files, isoforge=isoforge_run,
                     min_reads=1, verbose=False)
    assert view.annotation_gtf == isoforge_run + "_isoforge.gtf"
    assert view.polya_clusters and view.tss_clusters
    view.plot()
    rows = [t for t in view.canvas.of_type(TranscriptTrack) if type(t) is TranscriptTrack]
    assert sorted(t.label for t in rows) == ["TX_LONG", "TX_SHORT"]


def test_an_explicit_file_takes_precedence_over_the_run(isoforge_run, bam_files, gtf_path):
    view = IsoTracks(gene_name=GENE, bam_files=bam_files, isoforge=isoforge_run,
                     annotation_gtf=gtf_path, verbose=False)
    assert view.annotation_gtf == gtf_path


def test_isoform_rows_come_from_the_gtf_without_a_bed(view_kwargs):
    view = IsoTracks(**{**view_kwargs, "bed_file": None})
    view.plot()
    rows = [t for t in view.canvas.of_type(TranscriptTrack) if type(t) is TranscriptTrack]
    assert sorted(t.label for t in rows) == ["TX_LONG", "TX_SHORT"]
    exons = [c for f in rows[0].region_data for c in f.children if c.type == "exon"]
    assert exons, "the model has its exons"


def test_the_cli_takes_an_isoforge_prefix(isoforge_run, bam_files, tmp_path):
    bam = ";".join(f"{c}:{','.join(p)}" for c, p in bam_files.items())
    output = tmp_path / "gene.png"
    assert main(["plot", "--gene", GENE, "--isoforge", isoforge_run, "--bam", bam,
                 "--min-reads", "1", "--output", str(output), "--quiet"]) == 0
    assert output.stat().st_size > 1_000


def test_a_fusion_is_drawn_from_the_prefix(fusion_tables, tmp_path):
    view = FusionView("M000001", isoforge=fusion_tables, verbose=False)
    view.plot()
    assert view.fusion_id == "M000001"
    output = tmp_path / "fusion.png"
    assert main(["plot", "--fusion", "M000001", "--isoforge", fusion_tables,
                 "--output", str(output), "--quiet"]) == 0
    assert output.stat().st_size > 1_000


@pytest.fixture
def isoforge_run_with_alignments(isoforge_run, bam_files):
    """The run as ``isoforge annotate --write_bam`` leaves it, with its BAMs."""
    directory = isoforge_run + "_alignments"
    os.makedirs(directory)
    for paths in bam_files.values():
        for path in paths:
            shutil.copy(path, directory)
            shutil.copy(path + ".bai", directory)
    return isoforge_run


def test_bams_are_grouped_by_the_name_up_to_the_first_underscore():
    from isotracks.isoforge import group_by_condition

    grouped = group_by_condition(["x/Treated_1.bam", "x/Control_2.bam", "x/Control_1.bam",
                                  "x/Single.bam"])
    assert grouped == {"Control": ["x/Control_1.bam", "x/Control_2.bam"],
                       "Single": ["x/Single.bam"], "Treated": ["x/Treated_1.bam"]}


def test_the_runs_alignments_and_cds_bed_are_found(isoforge_run_with_alignments, bed_path):
    shutil.copy(bed_path, isoforge_run_with_alignments + "_isoforge_sqanti3.bed")
    found = find_outputs(isoforge_run_with_alignments)
    assert found["bed_file"].endswith("_isoforge_sqanti3.bed")
    assert sorted(found["bam_files"]) == ["Ctrl", "Trt"]
    assert [os.path.basename(p) for p in found["bam_files"]["Ctrl"]] == [
        "Ctrl_rep1.bam", "Ctrl_rep2.bam"]


def test_a_gene_is_drawn_from_the_prefix_alone(isoforge_run_with_alignments, tmp_path):
    view = IsoTracks(gene_name=GENE, isoforge=isoforge_run_with_alignments, min_reads=1,
                     verbose=False)
    assert sorted(view.bam_files) == ["Ctrl", "Trt"]
    output = tmp_path / "gene.png"
    assert main(["plot", "--gene", GENE, "--isoforge", isoforge_run_with_alignments,
                 "--min-reads", "1", "--output", str(output), "--quiet"]) == 0
    assert output.stat().st_size > 1_000


def test_given_bams_take_precedence_over_the_runs(isoforge_run_with_alignments, bam_files):
    view = IsoTracks(gene_name=GENE, isoforge=isoforge_run_with_alignments,
                     bam_files={"Only": bam_files["Ctrl"]}, verbose=False)
    assert list(view.bam_files) == ["Only"]


def test_a_run_without_alignments_needs_bams(isoforge_run, tmp_path):
    with pytest.raises(SystemExit, match="no BAM files"):
        main(["plot", "--gene", GENE, "--isoforge", isoforge_run,
              "--output", str(tmp_path / "x.png"), "--quiet"])
