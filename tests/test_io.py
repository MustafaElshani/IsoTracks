"""Readers: BED12 splitting, read blocks, coverage, poly(A) tails, config merge."""

from __future__ import annotations

import pandas as pd
import pysam
import pytest

from isotracks._dictutil import deep_copy_config, update_dict
from isotracks.io.alignments import CoverageReader, read_blocks, read_introns
from isotracks.io.features import BedTranscripts, Feature, FeatureArray
from isotracks.io.longreads import LongReadBam, read_key
from isotracks.io.clusters import ClusterSiteReader

from .conftest import CHROM


# ----------------------------------------------------------------- dict merge

def test_dotted_keys_reach_nested_values():
    config = {"style": {"exon": {"facecolor": "black"}}}
    update_dict(config, {"style.exon.facecolor": "red"})
    assert config["style"]["exon"]["facecolor"] == "red"


def test_nested_dicts_merge_rather_than_replace():
    config = {"style": {"exon": {"facecolor": "black", "zorder": 10}}}
    update_dict(config, {"style": {"exon": {"facecolor": "red"}}})
    assert config["style"]["exon"] == {"facecolor": "red", "zorder": 10}


def test_an_empty_dict_clears_a_branch():
    config = {"style": {"exon": {"facecolor": "black"}}}
    update_dict(config, {"style": {}})
    assert config["style"] == {}


def test_lists_are_copied_not_aliased():
    source = [1, 2, 3]
    config = {}
    update_dict(config, {"values": source})
    config["values"].append(4)
    assert source == [1, 2, 3]


def test_deep_copy_isolates_nested_edits():
    original = {"a": {"b": [1, 2]}}
    copy = deep_copy_config(original)
    copy["a"]["b"].append(3)
    assert original["a"]["b"] == [1, 2]


# -------------------------------------------------------------------- BED12

def test_bed12_splits_blocks_into_utr_and_cds(bed_path):
    features = BedTranscripts(file_name=bed_path).features
    long_tx = next(f for f in features if f.id == "TX_LONG")
    kinds = {child.type for child in long_tx.children}
    assert {"5UTR", "CDS", "3UTR", "intron"} <= kinds


def test_bed12_marks_the_terminal_exons(bed_path):
    long_tx = next(f for f in BedTranscripts(file_name=bed_path).features
                   if f.id == "TX_LONG")
    bodies = sorted((c for c in long_tx.children if c.type != "intron"),
                    key=lambda c: c.start)
    assert bodies[0].parm.get("is_first")
    assert bodies[-1].parm.get("is_last")


def test_a_noncoding_transcript_is_all_thin(tmp_path):
    path = tmp_path / "nc.bed"
    path.write_text("\t".join([
        "chr1", "100", "500", "NC1", "0", "+", "100", "100",
        "0", "2", "100,100,", "0,300,",
    ]) + "\n")
    feature = BedTranscripts(file_name=str(path)).features[0]
    assert {c.type for c in feature.children} == {"5UTR", "intron"}


def test_introns_fill_the_gaps_between_blocks(bed_path):
    long_tx = next(f for f in BedTranscripts(file_name=bed_path).features
                   if f.id == "TX_LONG")
    introns = sorted((c for c in long_tx.children if c.type == "intron"),
                     key=lambda c: c.start)
    assert [(i.start, i.end) for i in introns] == [(1_301, 2_999), (3_201, 5_999)]


def test_region_selection_keeps_overlapping_features(bed_path):
    array = BedTranscripts(file_name=bed_path)
    array.load_region_data((CHROM, 3_300, 3_500, "+"))
    assert len(array.region_data) == 2


def test_region_selection_subset_keeps_its_class(bed_path):
    array = BedTranscripts(file_name=bed_path)
    array.load_region_data((CHROM, 1_000, 2_000, "+"))
    assert isinstance(array.region_data, BedTranscripts)
    assert array.region_data.header == array.header


def test_feature_array_pushes_columns_back_onto_features():
    features = [Feature(1, 10, "chr1", "+", "a"), Feature(20, 30, "chr1", "+", "b")]
    array = FeatureArray(features)
    array.df["y_min"] = [0.0, 1.0]
    array.df["y_max"] = [1.0, 2.0]
    array.push_columns(["y_min", "y_max"])
    assert features[1].y_min == 1.0


# ------------------------------------------------------------------ BAM reads

def _read_with_cigar(cigar, start=1000):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": 20_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.reference_id = 0
    read.reference_start = start - 1
    read.cigartuples = cigar
    return read


def test_blocks_split_only_at_splice_junctions():
    read = _read_with_cigar([(0, 100), (1, 5), (0, 50), (2, 3), (0, 20),
                             (3, 500), (0, 80)])
    blocks = read_blocks(read)
    assert len(blocks) == 2
    assert blocks[0] == (1000, 1172)  # 100 + 50 + 3(del) + 20
    assert blocks[1] == (1673, 1752)


def test_soft_clips_do_not_extend_a_block():
    unclipped = read_blocks(_read_with_cigar([(0, 100)]))
    clipped = read_blocks(_read_with_cigar([(4, 30), (0, 100), (4, 20)]))
    assert unclipped == clipped


def test_introns_are_reported_as_spans_and_as_junctions():
    read = _read_with_cigar([(0, 100), (3, 500), (0, 80)])
    assert read_introns(read) == [(1100, 1599)]
    assert read_introns(read, as_junction=True) == [(1099, 1600)]


def test_coverage_counts_depth_and_junctions(bam_files):
    reader = CoverageReader(file_names=bam_files["Ctrl"])
    depth, junctions = reader.load_region_data((CHROM, 900, 7_000, "+"))
    assert depth["depth"].max() > 0
    assert not junctions.empty
    assert (junctions["ratio"] <= 2.0).all()


def test_junction_ratio_is_finite_at_zero_depth():
    junctions = pd.DataFrame({"start": [10], "end": [20], "count": [5]})
    result = CoverageReader._attach_depth(junctions, pd.DataFrame(columns=["pos", "depth"]))
    assert result["ratio"].iloc[0] == 0.0


def test_mapping_quality_zero_reads_are_dropped(bam_files):
    reader = CoverageReader(file_names=bam_files["Ctrl"], min_mapping_quality=61)
    depth, _ = reader.load_region_data((CHROM, 900, 7_000, "+"))
    assert depth["depth"].max() == 0


# -------------------------------------------------------------- long reads

def test_read_key_drops_the_pipeline_prefix():
    assert read_key("107:838|abc-123") == "abc-123"
    assert read_key("abc-123") == "abc-123"


def test_allowed_names_match_when_the_bam_carries_a_prefix(bam_files):
    reader = LongReadBam(
        file_names=bam_files["Ctrl"], allowed_read_names={"TX_LONG_read0"},
    )
    features = reader.load_region_data((CHROM, 900, 7_000, "+"))
    assert len(features) == 2  # one per replicate, matched on the read id


def test_allowed_names_match_when_the_prefixes_disagree(bam_files):
    reader = LongReadBam(
        file_names=bam_files["Ctrl"],
        allowed_read_names={"99:1234|TX_LONG_read0"},  # a different prefix
    )
    features = reader.load_region_data((CHROM, 900, 7_000, "+"))
    assert len(features) == 2


def test_polya_fallback_matches_across_differing_prefixes(bam_files):
    reader = LongReadBam(
        file_names=bam_files["Ctrl"][:1],
        allowed_read_names={"TX_LONG_read0"},
        polya_lengths={"77:5|TX_LONG_read0": 321},
    )
    features = reader.load_region_data((CHROM, 900, 7_000, "+"))
    assert features.df["polya_length"].iloc[0] == 60


def test_tail_length_comes_from_the_pt_tag(bam_files):
    reader = LongReadBam(file_names=bam_files["Trt"][:1])
    features = reader.load_region_data((CHROM, 900, 7_500, "+"))
    assert features.df["polya_length"].max() == 180 + 5 * 10


def test_tail_length_falls_back_to_a_supplied_table(tmp_path):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": 20_000}]}
    path = tmp_path / "untagged.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = "r1"
        read.query_sequence = "A" * 100
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 999
        read.mapping_quality = 60
        read.cigartuples = [(0, 100)]
        out.write(read)
    pysam.index(str(path))

    reader = LongReadBam(file_names=[str(path)], polya_lengths={"r1": 123})
    features = reader.load_region_data((CHROM, 900, 2_000, "+"))
    assert features.df["polya_length"].iloc[0] == 123


def test_polya_scale_only_changes_the_drawing(bam_files):
    honest = LongReadBam(file_names=bam_files["Trt"][:1], polya_scale=1.0)
    honest_features = honest.load_region_data((CHROM, 900, 9_000, "+"))
    exaggerated = LongReadBam(file_names=bam_files["Trt"][:1], polya_scale=4.0)
    big_features = exaggerated.load_region_data((CHROM, 900, 9_000, "+"))

    assert honest.polya_scale != exaggerated.polya_scale
    assert (honest_features.df["polya_length"].sort_values().values
            == big_features.df["polya_length"].sort_values().values).all()
    assert big_features.df["end"].max() > honest_features.df["end"].max()


def test_a_minus_strand_tail_extends_the_start(tmp_path):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": 20_000}]}
    path = tmp_path / "minus.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = "r1"
        read.query_sequence = "A" * 100
        read.flag = 16  # reverse
        read.reference_id = 0
        read.reference_start = 4_999
        read.mapping_quality = 60
        read.cigartuples = [(0, 100)]
        read.set_tag("pt", 80, value_type="i")
        out.write(read)
    pysam.index(str(path))

    features = LongReadBam(file_names=[str(path)]).load_region_data(
        (CHROM, 4_000, 6_000, "-"))
    feature = features.features[0]
    assert feature.start == 5_000 - 80
    assert feature.children[0].type == "polyA"


def test_max_polya_length_ignores_filtered_reads(bam_files):
    reader = LongReadBam(file_names=bam_files["Trt"], allowed_read_names={"nothing"})
    assert reader.max_polya_length((CHROM, 900, 7_500, "+")) == 0


# ----------------------------------------------------------- cluster sites

def test_cluster_reader_parses_position_at_motif(polya_bed_path):
    sites = ClusterSiteReader(polya_bed_path).read_region((CHROM, 1, 10_000))
    assert set(sites["motif"]) == {"AATAAA", "ATTAAA"}
    assert sites.loc[sites["site"] == 6_800, "motif_pos"].iloc[0] == 6_815


def test_cluster_reader_parses_a_bare_motif(tmp_path):
    path = tmp_path / "plain.bed"
    path.write_text("\t".join([
        CHROM, "100", "120", "110", "9", "+", "utr", "G",
        "1.0", "9", "0.5", "2", "AATAAA", ".",
    ]) + "\n")
    sites = ClusterSiteReader(str(path)).read_region((CHROM, 1, 1_000))
    assert sites["motif"].iloc[0] == "AATAAA"
    assert sites["motif_pos"].iloc[0] == 110  # falls back to the site


def test_cluster_reader_reads_tss_clusters_without_a_motif(tmp_path):
    path = tmp_path / "tss.bed"
    path.write_text("\t".join([
        CHROM, "900", "920", "910", "71", "+", "exon", "G",
        "1.75", "71", "1.0000", ".",
    ]) + "\n")
    sites = ClusterSiteReader(str(path)).read_region((CHROM, 1, 2_000))
    assert list(sites["site"]) == [910]
    assert sites["motif"].iloc[0] is None


def test_require_motif_drops_clusters_without_one(tmp_path):
    path = tmp_path / "none.bed"
    path.write_text("\t".join([
        CHROM, "100", "120", "110", "9", "+", "utr", "G",
        "1.0", "9", "0.5", "2", "None", ".",
    ]) + "\n")
    assert ClusterSiteReader(str(path), require_motif=True).read_region(
        (CHROM, 1, 1_000)).empty
    assert not ClusterSiteReader(str(path), require_motif=False).read_region(
        (CHROM, 1, 1_000)).empty


def test_cluster_reader_respects_the_region(polya_bed_path):
    sites = ClusterSiteReader(polya_bed_path).read_region((CHROM, 1, 4_000))
    assert list(sites["site"]) == [3_600]


# ------------------------------------------------------- read-end density

def _pool(paths, strand="+"):
    from isotracks.io.longreads import RegionReadPool
    return RegionReadPool(paths, strand_filter=strand)


def test_end_density_peaks_where_reads_start_and_stop(bam_files):
    from isotracks.io.ends import ReadEndReader

    reader = ReadEndReader([_pool(bam_files["Ctrl"])], bandwidth=10)
    density = reader.read_region((CHROM, 900, 7_000, "+"))

    def peaks(values, positions, floor=0.25):
        """Local maxima above a fraction of the highest point."""
        cut = values.max() * floor
        return [
            int(positions[i]) for i in range(1, len(values) - 1)
            if values[i] >= cut and values[i] >= values[i - 1] and values[i] > values[i + 1]
        ]

    five = peaks(density.five_prime, density.positions)
    three = peaks(density.three_prime, density.positions)
    assert len(five) == 1 and abs(five[0] - 1_000) <= 15  # both isoforms share a TSS
    assert len(three) == 2  # but end in different places
    assert min(abs(p - 3_600) for p in three) <= 15
    assert min(abs(p - 6_800) for p in three) <= 15


def test_end_density_counts_every_read_once_at_each_end(bam_files):
    from isotracks.io.ends import ReadEndReader

    reader = ReadEndReader([_pool(bam_files["Ctrl"])])
    density = reader.read_region((CHROM, 900, 7_000, "+"))
    assert density.five_prime_reads == 24  # 2 isoforms x 6 reads x 2 replicates
    assert density.three_prime_reads == 24


def test_end_density_respects_the_allowed_reads(bam_files):
    from isotracks.io.ends import ReadEndReader

    reader = ReadEndReader([_pool(bam_files["Ctrl"])],
                           allowed_read_names={"TX_LONG_read0"})
    density = reader.read_region((CHROM, 900, 7_000, "+"))
    assert density.five_prime_reads == 2  # one per replicate
    assert not density.is_empty()


def test_end_density_is_empty_when_nothing_matches(bam_files):
    from isotracks.io.ends import ReadEndReader

    reader = ReadEndReader([_pool(bam_files["Ctrl"])], allowed_read_names={"nothing"})
    assert reader.read_region((CHROM, 900, 7_000, "+")).is_empty()


def test_a_reverse_read_has_its_five_prime_end_at_the_higher_coordinate(tmp_path):
    from isotracks.io.ends import ReadEndReader

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": 20_000}]}
    path = tmp_path / "minus.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = "r1"
        read.query_sequence = "A" * 200
        read.flag = 16  # reverse
        read.reference_id = 0
        read.reference_start = 4_999  # spans 5,000..5,199
        read.mapping_quality = 60
        read.cigartuples = [(0, 200)]
        out.write(read)
    pysam.index(str(path))

    reader = ReadEndReader([_pool([str(path)], strand="-")], bandwidth=1)
    density = reader.read_region((CHROM, 4_800, 5_400, "-"))
    five_peak = int(density.positions[density.five_prime.argmax()])
    three_peak = int(density.positions[density.three_prime.argmax()])
    assert abs(five_peak - 5_199) <= 2
    assert abs(three_peak - 5_000) <= 2


# --------------------------------------------------------- modification calls

def test_mod_labels_name_the_dorado_codes():
    from isotracks.io.modifications import mod_label

    assert mod_label("a") == "m6A"
    assert mod_label("17802") == "pseU"
    assert mod_label("19229") == "Gm"
    assert mod_label("99999") == "99999"  # unknown codes show as themselves


def test_pileup_pools_replicates_by_summing_counts(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    reader = ModPileupReader(modkit_pileups["Trt"], codes=["a"],
                             min_coverage=10, min_fraction=0.3)
    sites = reader.read_region((CHROM, 6_000, 6_600, "+"))
    row = sites.frame[sites.frame["position"] == 6_100].iloc[0]
    assert row["coverage"] == 20  # 10 + 10 across the two replicates
    assert row["modified"] == 16  # 8 + 8
    assert row["fraction"] == pytest.approx(0.8)


def test_pileup_drops_thinly_covered_sites(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    reader = ModPileupReader(modkit_pileups["Trt"], codes=["a"],
                             min_coverage=10, min_fraction=0.0)
    sites = reader.read_region((CHROM, 6_000, 6_600, "+"))
    assert 6_300 not in set(sites.frame["position"])  # only 4 reads pooled


def test_pileup_drops_barely_modified_sites_but_counts_them(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    reader = ModPileupReader(modkit_pileups["Trt"], codes=["a"],
                             min_coverage=10, min_fraction=0.3)
    sites = reader.read_region((CHROM, 6_000, 6_600, "+"))
    assert 6_400 not in set(sites.frame["position"])  # 5% modified
    assert sites.assessed > len(sites.frame)


def test_pileup_keeps_only_the_requested_codes(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    reader = ModPileupReader(modkit_pileups["Trt"], codes=["a", "17802"],
                             min_coverage=10, min_fraction=0.3)
    sites = reader.read_region((CHROM, 6_000, 6_600, "+"))
    assert set(sites.codes()) == {"a", "17802"}  # not the 5mC site


def test_pileup_separates_the_conditions(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    def fraction(condition):
        reader = ModPileupReader(modkit_pileups[condition], codes=["a"],
                                 min_coverage=10, min_fraction=0.0)
        frame = reader.read_region((CHROM, 6_000, 6_600, "+")).frame
        return float(frame[frame["position"] == 6_100]["fraction"].iloc[0])

    assert fraction("Ctrl") == pytest.approx(0.1)
    assert fraction("Trt") == pytest.approx(0.8)


def test_pileup_respects_the_strand_filter(modkit_pileups):
    from isotracks.io.modifications import ModPileupReader

    reader = ModPileupReader(modkit_pileups["Trt"], codes=["a"],
                             min_coverage=10, min_fraction=0.0, strand_filter="-")
    assert reader.read_region((CHROM, 6_000, 6_600, "-")).is_empty()


def test_an_unindexed_pileup_says_how_to_index_it(tmp_path):
    from isotracks.io.modifications import ModkitInputError, check_pileups

    plain = tmp_path / "pileup.bed.gz"
    plain.write_bytes(b"")
    with pytest.raises(ModkitInputError, match="tabix"):
        check_pileups({"Ctrl": [str(plain)]})


def test_a_missing_pileup_is_reported(tmp_path):
    from isotracks.io.modifications import ModkitInputError, check_pileups

    with pytest.raises(ModkitInputError, match="not found"):
        check_pileups({"Ctrl": [str(tmp_path / "nope.bed.gz")]})


# ------------------------------------------------- differential modification

def test_dmr_reader_keeps_only_significant_sizeable_changes(mod_diff_table):
    from isotracks.io.moddiff import ModDiffReader

    reader = ModDiffReader(mod_diff_table, min_coverage=10,
                           max_p_value=0.05, min_effect=0.2)
    sites = reader.read_region((CHROM, 6_000, 6_400, "+"))
    assert list(sites.frame["position"]) == [6_100, 6_150]
    assert sites.tested == 4  # the 4-read site never counted as tested


def test_dmr_effect_is_positive_when_the_modification_is_lost(mod_diff_table):
    from isotracks.io.moddiff import ModDiffReader

    reader = ModDiffReader(mod_diff_table, min_coverage=10,
                           max_p_value=0.05, min_effect=0.2)
    sites = reader.read_region((CHROM, 6_000, 6_400, "+"))
    assert list(sites.lost["position"]) == [6_100]
    assert list(sites.gained["position"]) == [6_150]
    assert sites.frame.set_index("position").loc[6_100, "effect_size"] > 0


def test_dmr_reader_respects_the_strand_filter(mod_diff_table):
    from isotracks.io.moddiff import ModDiffReader

    reader = ModDiffReader(mod_diff_table, min_coverage=10, strand_filter="-")
    assert reader.read_region((CHROM, 6_000, 6_400, "-")).is_empty()


def test_dmr_balanced_uses_the_balanced_columns(mod_diff_table):
    from isotracks.io.moddiff import ModDiffReader

    plain = ModDiffReader(mod_diff_table, min_coverage=10).read_region(
        (CHROM, 6_000, 6_400, "+"))
    balanced = ModDiffReader(mod_diff_table, min_coverage=10, balanced=True).read_region(
        (CHROM, 6_000, 6_400, "+"))
    # The fixture writes the same values into both, so they must agree.
    assert list(plain.frame["effect_size"]) == list(balanced.frame["effect_size"])


def test_an_unindexed_dmr_table_says_how_to_index_it(tmp_path):
    from isotracks.io.moddiff import ModDiffInputError, check_diff_tables

    path = tmp_path / "sites.bed.gz"
    path.write_bytes(b"")
    with pytest.raises(ModDiffInputError, match="tabix"):
        check_diff_tables({"A vs B": str(path)})
