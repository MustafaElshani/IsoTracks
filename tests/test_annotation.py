"""GTF parsing, read-annotation loading, BED12 selection, sample mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from isotracks.annotation import (
    GtfIndex,
    ReadAnnotation,
    canonical_columns,
    load_bed,
    map_samples_to_conditions,
    parse_gtf_attributes,
    select_bed_transcripts,
)

from .conftest import CHROM, GENE


def test_parse_gtf_attributes_strips_quotes():
    attrs = parse_gtf_attributes('gene_id "G1"; gene_name "TESTG"; level 2;')
    assert attrs == {"gene_id": "G1", "gene_name": "TESTG", "level": "2"}


def test_gene_records_collect_gene_transcripts_and_exons(gtf_path):
    records = GtfIndex(gtf_path).gene_records(GENE)
    assert records.gene.chromosome == CHROM
    assert set(records.transcripts["transcript_id"]) == {"TX_LONG", "TX_SHORT"}
    assert len(records.exons["TX_LONG"]) == 3


def test_missing_gene_yields_empty_records(gtf_path):
    assert GtfIndex(gtf_path).gene_records("NOT_A_GENE").is_empty


def test_transcript_extent_spans_all_transcripts(gtf_path):
    extent = GtfIndex(gtf_path).gene_records(GENE).transcript_extent()
    assert (extent.start, extent.end) == (1_000, 6_800)


def test_exons_are_numbered_five_to_three_on_minus_strand(tmp_path):
    gtf = tmp_path / "minus.gtf"
    attrs = 'gene_id "G"; gene_name "M"; transcript_id "T1";'
    gtf.write_text("\n".join(
        "\t".join(["chr1", "t", feature, str(start), str(end), ".", "-", ".", attrs])
        for feature, start, end in [
            ("transcript", 100, 900), ("exon", 100, 200), ("exon", 800, 900),
        ]
    ) + "\n")

    records = GtfIndex(str(gtf)).gene_records("M")
    assert records.exons_in_transcription_order()["T1"] == [(800, 900), (100, 200)]


def test_gtf_scan_is_cached(gtf_path, monkeypatch):
    index = GtfIndex(gtf_path)
    index.gene_records(GENE)

    def fail(*_args, **_kwargs):
        raise AssertionError("cached gene should not be re-scanned")

    monkeypatch.setattr(GtfIndex, "_scan", fail)
    assert GtfIndex(gtf_path).gene_records(GENE).gene is not None


def test_canonical_columns_maps_isoforge_spellings():
    frame = pd.DataFrame(columns=["read_name", "annot_transcript_id", "dataset",
                                  "chromosome", "strand", "annot_gene_name"])
    renamed = canonical_columns(frame)
    assert {"transcript_id", "sample", "gene_name"} <= set(renamed.columns)


def test_canonical_columns_prefers_the_plain_transcript_id():
    frame = pd.DataFrame(columns=["transcript_id", "annot_transcript_id"])
    renamed = canonical_columns(frame)
    assert "annot_transcript_id" in renamed.columns  # left alone, not clobbered


def test_read_annotation_groups_reads_by_transcript(read_annot_path):
    reads = ReadAnnotation(read_annot_path).for_gene(GENE, CHROM, "+")
    assert reads.counts() == {"TX_LONG": 24, "TX_SHORT": 24}
    assert len(reads.read_names) == 48


def test_read_annotation_filters_by_chromosome(read_annot_path):
    reads = ReadAnnotation(read_annot_path).for_gene(GENE, "chrOTHER", "+")
    assert reads.counts() == {}


def test_counts_by_condition_sums_over_replicates(read_annot_path):
    reads = ReadAnnotation(read_annot_path).for_gene(GENE, CHROM, "+")
    mapping = {"Ctrl_rep1": "Ctrl", "Ctrl_rep2": "Ctrl",
               "Trt_rep1": "Trt", "Trt_rep2": "Trt"}
    table = reads.counts_by_condition(mapping)
    long_ctrl = table.query("transcript_id == 'TX_LONG' and condition == 'Ctrl'")
    assert int(long_ctrl["count"].iloc[0]) == 12


def test_read_annotation_rejects_a_table_without_gene_names(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("read_name\ttranscript_id\nr1\tT1\n")
    with pytest.raises(ValueError, match="gene-name column"):
        ReadAnnotation(str(path)).for_gene(GENE)


def test_parquet_and_tsv_agree(read_annot_path, tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.read_csv(read_annot_path, sep="\t")
    parquet = tmp_path / "read_annot.parquet"
    frame.to_parquet(parquet)

    from_tsv = ReadAnnotation(read_annot_path).for_gene(GENE, CHROM, "+")
    from_parquet = ReadAnnotation(str(parquet)).for_gene(GENE, CHROM, "+")
    assert from_tsv.counts() == from_parquet.counts()


def test_bed12_selection_by_transcript_id(bed_path):
    selected = select_bed_transcripts(load_bed(bed_path), ["TX_LONG"])
    assert list(selected["name"]) == ["TX_LONG"]


def test_bed12_selection_falls_back_to_transcript_names(tmp_path):
    path = tmp_path / "named.bed"
    path.write_text("\t".join([
        "chr1", "100", "500", "TESTG-201", "0", "+", "100", "500",
        "0", "1", "400,", "0,",
    ]) + "\n")

    selected = select_bed_transcripts(
        load_bed(str(path)), ["ENST001"], name_map={"ENST001": "TESTG-201"},
    )
    assert list(selected["name"]) == ["ENST001"]  # rewritten to the id


def test_sample_mapping_prefers_the_longest_match():
    mapping = map_samples_to_conditions(
        {"Ctrl": ["/x/S_Bio1_LR_gnome_sorted.bam"], "Trt": ["/x/S_Bio11_LR_gnome_sorted.bam"]},
        known_samples=["S_Bio1_LR", "S_Bio11_LR"],
    )
    assert mapping == {"S_Bio1_LR": "Ctrl", "S_Bio11_LR": "Trt"}


def test_sample_mapping_ignores_unknown_bams():
    assert map_samples_to_conditions({"Ctrl": ["/x/other.bam"]}, ["sampleA"]) == {}


# --------------------------------------- IsoForge Parquet vs its TSV

def test_the_parquet_holds_reads_the_tsv_would_not(isoforge_parquet):
    from isotracks.annotation import ALL_READS, ReadAnnotation

    reads = ReadAnnotation(isoforge_parquet["parquet"], read_filter=ALL_READS,
                           verbose=False).for_gene(GENE)
    assert sum(reads.counts().values()) == isoforge_parquet["rows"]
    assert set(reads.counts()) == {"TX_LONG", "TX_SHORT"}


def test_restrict_to_drops_transcripts_the_gtf_does_not_carry(isoforge_parquet):
    from isotracks.annotation import ALL_READS, ReadAnnotation

    reads = ReadAnnotation(isoforge_parquet["parquet"], read_filter=ALL_READS,
                           verbose=False).for_gene(GENE)
    assert set(reads.counts()) == {"TX_LONG", "TX_SHORT"}

    kept = reads.restrict_to({"TX_LONG"})
    assert set(kept.counts()) == {"TX_LONG"}
    assert len(kept.frame) < len(reads.frame)


def test_restrict_to_is_a_no_op_without_a_transcript_set(isoforge_parquet):
    from isotracks.annotation import ALL_READS, ReadAnnotation

    reads = ReadAnnotation(isoforge_parquet["parquet"], read_filter=ALL_READS,
                           verbose=False).for_gene(GENE)
    assert reads.restrict_to(set()) is reads


def test_supported_is_the_default_and_drops_failed_end_validation(isoforge_parquet):
    from isotracks.annotation import ReadAnnotation

    everything = ReadAnnotation(isoforge_parquet["parquet"], read_filter="all",
                                verbose=False).for_gene(GENE)
    supported = ReadAnnotation(isoforge_parquet["parquet"],
                               verbose=False).for_gene(GENE)  # default
    assert supported.counts()["TX_SHORT"] == everything.counts()["TX_SHORT"] - 4
    assert supported.counts()["TX_LONG"] == everything.counts()["TX_LONG"]


def test_the_internal_priming_rule_is_off_unless_asked_for(isoforge_parquet):
    from isotracks.annotation import ReadAnnotation

    default = ReadAnnotation(isoforge_parquet["parquet"], verbose=False).for_gene(GENE)
    strict = ReadAnnotation(isoforge_parquet["parquet"], verbose=False,
                            max_frac_a=0.01).for_gene(GENE)
    # Every fixture read has a poly(A) signal, so the rule spares them all.
    assert sum(strict.counts().values()) == sum(default.counts().values())


def test_an_unknown_read_filter_is_refused(isoforge_parquet):
    from isotracks.annotation import ReadAnnotation

    with pytest.raises(ValueError, match="unknown read filter"):
        ReadAnnotation(isoforge_parquet["parquet"], read_filter="most",
                       verbose=False).for_gene(GENE)


def test_the_representative_isoform_is_mane_then_canonical_then_longest():
    import pandas as pd
    from isotracks.annotation import GeneRecords

    exons = {"long": [(1, 5_000)], "canon": [(1, 300)], "mane": [(1, 100)]}
    frame = pd.DataFrame({"transcript_id": ["long", "canon", "mane"],
                          "mane": [False, False, True],
                          "canonical": [False, True, True]})
    assert GeneRecords("G", transcripts=frame, exons=exons).representative()[0] == "mane"
    frame["mane"] = False
    assert GeneRecords("G", transcripts=frame, exons=exons).representative()[0] == "canon"
    frame["canonical"] = False
    assert GeneRecords("G", transcripts=frame, exons=exons).representative() == (
        "long", [(1, 5_000)])
    assert GeneRecords("G").representative() == (None, [])


def test_gencode_tags_are_read_though_the_key_repeats(tmp_path):
    from isotracks.annotation import GtfIndex

    attrs = ('gene_id "G1"; transcript_id "T1"; gene_name "G"; tag "basic"; '
             'tag "Ensembl_canonical"; tag "MANE_Select";')
    path = tmp_path / "tags.gtf"
    path.write_text(
        f"chr1\tsrc\ttranscript\t1\t500\t.\t+\t.\t{attrs}\n"
        f"chr1\tsrc\texon\t1\t500\t.\t+\t.\t{attrs}\n")
    row = GtfIndex(str(path)).gene_records("G").transcripts.iloc[0]
    assert bool(row["mane"]) and bool(row["canonical"])
