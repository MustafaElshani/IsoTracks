"""Synthetic fixtures: a two-isoform gene, two conditions, poly(A)-tagged reads."""

from __future__ import annotations

import matplotlib
import pysam
import pytest

matplotlib.use("Agg")

CHROM = "chrT"
CHROM_LENGTH = 20_000
GENE = "TESTG"

# Two isoforms sharing exon 1, differing in their 3' end.
ISOFORMS = {
    "TX_LONG": [(1_000, 1_300), (3_000, 3_200), (6_000, 6_800)],
    "TX_SHORT": [(1_000, 1_300), (3_000, 3_600)],
}
CODING = {"TX_LONG": (1_100, 6_400), "TX_SHORT": (1_100, 3_400)}


@pytest.fixture(autouse=True)
def _isolated_figures_and_theme():
    """Close every figure and put the theme back after each test."""
    import matplotlib.pyplot as plt
    from isotracks import theme as theme_module

    before = theme_module.active()
    yield
    plt.close("all")
    theme_module.use(before)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("isotracks_data")


@pytest.fixture(scope="session")
def gtf_path(data_dir):
    path = data_dir / "models.gtf"
    lines = []
    span = (min(s for spans in ISOFORMS.values() for s, _ in spans),
            max(e for spans in ISOFORMS.values() for _, e in spans))
    lines.append(_gtf_line("gene", span[0], span[1],
                           f'gene_id "G1"; gene_name "{GENE}";'))
    for tid, spans in ISOFORMS.items():
        start, end = spans[0][0], spans[-1][1]
        attrs = (f'gene_id "G1"; gene_name "{GENE}"; transcript_id "{tid}"; '
                 f'transcript_name "{GENE}-{tid}"; '
                 f'tss_site "{start}"; polyA_site "{end}";')
        lines.append(_gtf_line("transcript", start, end, attrs))
        for exon_start, exon_end in spans:
            lines.append(_gtf_line("exon", exon_start, exon_end, attrs))
    path.write_text("#!genome-build test\n" + "\n".join(lines) + "\n")
    return str(path)


def _gtf_line(feature, start, end, attributes):
    return "\t".join([CHROM, "test", feature, str(start), str(end), ".", "+", ".", attributes])


@pytest.fixture(scope="session")
def bed_path(data_dir):
    path = data_dir / "models.bed"
    rows = []
    for tid, spans in ISOFORMS.items():
        start, end = spans[0][0] - 1, spans[-1][1]
        sizes = ",".join(str(e - s + 1) for s, e in spans) + ","
        offsets = ",".join(str(s - 1 - start) for s, _ in spans) + ","
        thick_start, thick_end = CODING[tid]
        rows.append("\t".join([
            CHROM, str(start), str(end), tid, "0", "+",
            str(thick_start - 1), str(thick_end), "0", str(len(spans)), sizes, offsets,
        ]))
    path.write_text("\n".join(rows) + "\n")
    return str(path)


@pytest.fixture(scope="session")
def tss_bed_path(data_dir):
    """A 12-column tss_clusters.bed, which carries no motif."""
    path = data_dir / "tss_clusters.bed"
    rows = [
        "\t".join([CHROM, "995", "1005", "1000", "90", "+", "exon",
                    "G1", "2.1", "90", "1.0000", "."]),
    ]
    path.write_text("\n".join(rows) + "\n")
    return str(path)


@pytest.fixture(scope="session")
def polya_bed_path(data_dir):
    """Two clusters, one at each isoform's 3' end."""
    path = data_dir / "polya_clusters.bed"
    rows = [
        "\t".join([CHROM, "6790", "6810", "6800", "120", "+", "three_prime_utr",
                   "G1", "9.5", "120", "0.8", "4", "6815@AATAAA", "6800"]),
        "\t".join([CHROM, "3590", "3610", "3600", "80", "+", "three_prime_utr",
                   "G1", "6.1", "80", "0.5", "3", "3615@ATTAAA", "3600"]),
    ]
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _write_bam(path, reads, reverse=False):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": CHROM_LENGTH}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, spans, polya, *strand in sorted(reads, key=lambda r: r[1][0][0]):
            read = _make_read(out.header, name, spans, polya)
            read.is_reverse = strand[0] if strand else reverse
            out.write(read)
    pysam.index(str(path))
    return str(path)


def _make_read(header, name, spans, polya):
    read = pysam.AlignedSegment(header)
    read.query_name = name
    length = sum(e - s + 1 for s, e in spans)
    read.query_sequence = "A" * length
    read.query_qualities = pysam.qualitystring_to_array("I" * length)
    read.flag = 0
    read.reference_id = 0
    read.reference_start = spans[0][0] - 1
    read.mapping_quality = 60

    cigar = []
    for index, (start, end) in enumerate(spans):
        if index:
            cigar.append((3, start - spans[index - 1][1] - 1))  # N
        cigar.append((0, end - start + 1))  # M
    read.cigartuples = cigar
    if polya is not None:
        read.set_tag("pt", int(polya), value_type="i")
    return read


@pytest.fixture(scope="session")
def bam_files(data_dir):
    """Two conditions x two replicates; treated reads carry longer tails."""
    files = {}
    for condition, tail_base in (("Ctrl", 60), ("Trt", 180)):
        paths = []
        for replicate in (1, 2):
            reads = []
            for tid, spans in ISOFORMS.items():
                for index in range(6):
                    reads.append((
                        f"{condition}_rep{replicate}|{tid}_read{index}",
                        spans,
                        tail_base + index * 10,
                    ))
            paths.append(_write_bam(
                data_dir / f"{condition}_rep{replicate}.bam", reads,
            ))
        files[condition] = paths
    return files


@pytest.fixture(scope="session")
def read_annot_path(data_dir, bam_files):
    """A read annotation matching the BAMs, in the TSV spelling."""
    path = data_dir / "read_annot.tsv"
    header = ["read_name", "sample", "Chromosome", "Strand",
              "annot_gene_name", "transcript_id", "annot_transcript_name"]
    rows = ["\t".join(header)]
    for condition in ("Ctrl", "Trt"):
        for replicate in (1, 2):
            sample = f"{condition}_rep{replicate}"
            for tid in ISOFORMS:
                for index in range(6):
                    rows.append("\t".join([
                        f"{sample}|{tid}_read{index}", sample, CHROM, "+",
                        GENE, tid, f"{GENE}-{tid}",
                    ]))
    path.write_text("\n".join(rows) + "\n")
    return str(path)


@pytest.fixture
def view_kwargs(gtf_path, bed_path, polya_bed_path, tss_bed_path,
                bam_files, read_annot_path):
    return dict(
        gene_name=GENE,
        bam_files=bam_files,
        annotation_gtf=gtf_path,
        read_annot=read_annot_path,
        bed_file=bed_path,
        polya_clusters=polya_bed_path,
        tss_clusters=tss_bed_path,
        min_reads=1,
        verbose=False,
    )


def _by_position(rows):
    """Sort BED rows by start, which is what tabix requires of any of these."""
    return sorted(rows, key=lambda row: int(row.split("\t")[1]))


def _bedmethyl_row(position, code, strand, coverage, modified):
    """One modkit bedMethyl line: 18 columns, 0-based start."""
    percent = 100.0 * modified / coverage if coverage else 0.0
    return "\t".join([
        CHROM, str(position - 1), str(position), code, str(coverage), strand,
        str(position - 1), str(position), "255,0,0",
        str(coverage), f"{percent:.2f}", str(modified), str(coverage - modified),
        "0", "0", "0", "0", "0",
    ])


@pytest.fixture(scope="session")
def modkit_pileups(data_dir):
    """Two replicates per condition, tabix-indexed, as modkit writes them."""
    files = {}
    for condition, modified in (("Ctrl", 1), ("Trt", 8)):
        paths = []
        for replicate in (1, 2):
            rows = [
                # A well-covered m6A site: the difference between the arms.
                _bedmethyl_row(6_100, "a", "+", 10, modified),
                # Pseudouridine, the same in both arms.
                _bedmethyl_row(6_200, "17802", "+", 10, 6),
                # Too few reads to mean anything.
                _bedmethyl_row(6_300, "a", "+", 2, 2),
                _bedmethyl_row(6_400, "a", "+", 20, 1),
                # A code the default selection does not ask for.
                _bedmethyl_row(6_500, "m", "+", 20, 18),
                _bedmethyl_row(4_500, "a", "+", 30, 24),
            ]
            plain = data_dir / f"{condition}_rep{replicate}_pileup.bed"
            plain.write_text("\n".join(_by_position(rows)) + "\n")
            packed = str(plain) + ".gz"
            pysam.tabix_compress(str(plain), packed, force=True)
            pysam.tabix_index(packed, preset="bed", force=True)
            paths.append(packed)
        files[condition] = paths
    return files


@pytest.fixture(scope="session")
def isoforge_parquet(data_dir, read_annot_path):
    """An IsoForge-style Parquet: every read, supported or not, filtered or not."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.read_csv(read_annot_path, sep="\t")
    frame = frame.rename(columns={"transcript_id": "annot_transcript_id",
                                  "sample": "dataset",
                                  "Chromosome": "chromosome", "Strand": "strand"})
    numeric = {tid: 100 + i for i, tid in enumerate(sorted(frame["annot_transcript_id"].unique()))}
    frame["transcript_ID"] = frame["annot_transcript_id"].map(numeric)

    frame["cluster_supported"] = True
    unsupported = frame.index[frame["annot_transcript_id"] == "TX_SHORT"][:4]
    frame.loc[unsupported, "cluster_supported"] = False

    frame["fraction_As"] = 0.05
    frame["polya_signal"] = "AATAAA"
    frame["polya_length"] = 100

    path = data_dir / "isoforge_read_annot.parquet"
    frame.to_parquet(path)

    whitelist = data_dir / "isoforge_filter.csv"
    whitelist.write_text(f"1,{numeric['TX_LONG']}\n")
    return {"parquet": str(path), "filter_csv": str(whitelist),
            "numeric": numeric, "rows": len(frame)}


def _dmr_row(position, strand, a_frac, b_frac, p_value, a_total=40, b_total=40):
    """One `modkit dmr pair` site line: 25 columns, 0-based start."""
    effect = a_frac - b_frac  # modkit's convention: a - b
    return "\t".join([
        CHROM, str(position - 1), str(position), ".", "0", strand,
        f"a:{int(a_frac * a_total)}", str(a_total),
        f"a:{int(b_frac * b_total)}", str(b_total),
        f"a:{a_frac * 100:.2f}", f"a:{b_frac * 100:.2f}",
        f"{a_frac}", f"{b_frac}",
        f"{p_value}", f"{effect}",
        f"{p_value}", f"{effect}",
        "100", "100", "-", "-", "0", "0", "0",
    ])


@pytest.fixture(scope="session")
def mod_diff_table(data_dir):
    """A dmr-pair table: one strong loss, one gain, three rejects, one intronic."""
    path = data_dir / "dmr_sites.bed"
    path.write_text("\n".join(_by_position([
        _dmr_row(6_100, "+", 0.95, 0.05, 1e-20),  # lost in b: kept
        _dmr_row(6_150, "+", 0.10, 0.80, 1e-18),  # gained in b: kept
        _dmr_row(6_200, "+", 0.50, 0.45, 1e-15),  # too small an effect
        _dmr_row(6_250, "+", 0.90, 0.10, 0.40),  # not significant
        _dmr_row(6_300, "+", 0.90, 0.10, 1e-15, a_total=4, b_total=4),  # too few reads
        _dmr_row(4_500, "+", 0.95, 0.05, 1e-20),  # intronic: no model covers it
    ])) + "\n")
    packed = str(path) + ".gz"
    pysam.tabix_compress(str(path), packed, force=True)
    pysam.tabix_index(packed, preset="bed", force=True)
    return packed


# ------------------------------------------------ a minus-strand gene

MINUS_GENE = "TESTM"
MINUS_ISOFORMS = {
    "TXM_LONG": [(10_000, 10_800), (13_000, 13_200), (15_000, 15_300)],
    "TXM_SHORT": [(11_400, 12_000), (15_000, 15_300)],
}
#: 1-based inclusive CDS; the 3' UTR is the stretch *below* it on this strand.
MINUS_CODING = {"TXM_LONG": (10_600, 15_200), "TXM_SHORT": (11_700, 15_200)}
MINUS_TSS, MINUS_POLYA = 15_300, 10_000


@pytest.fixture(scope="session")
def minus_view_kwargs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("isotracks_minus")
    attrs_for = lambda tid: (f'gene_id "G2"; gene_name "{MINUS_GENE}"; '
                             f'transcript_id "{tid}"; transcript_name "{MINUS_GENE}-{tid}";')

    def gtf_line(feature, start, end, attrs):
        return "\t".join([CHROM, "test", feature, str(start), str(end), ".", "-", ".", attrs])

    everything = [s for spans in MINUS_ISOFORMS.values() for s in spans]
    lines = [gtf_line("gene", min(a for a, _ in everything), max(b for _, b in everything),
                      f'gene_id "G2"; gene_name "{MINUS_GENE}";')]
    for tid, spans in MINUS_ISOFORMS.items():
        lines.append(gtf_line("transcript", spans[0][0], spans[-1][1], attrs_for(tid)))
        lines += [gtf_line("exon", a, b, attrs_for(tid)) for a, b in spans]
    gtf = directory / "minus.gtf"
    gtf.write_text("\n".join(lines) + "\n")

    rows = []
    for tid, spans in MINUS_ISOFORMS.items():
        start, end = spans[0][0] - 1, spans[-1][1]
        sizes = ",".join(str(b - a + 1) for a, b in spans) + ","
        offsets = ",".join(str(a - 1 - start) for a, _ in spans) + ","
        cds_start, cds_end = MINUS_CODING[tid]
        rows.append("\t".join([CHROM, str(start), str(end), tid, "0", "-",
                               str(cds_start - 1), str(cds_end), "0",
                               str(len(spans)), sizes, offsets]))
    bed = directory / "minus.bed"
    bed.write_text("\n".join(rows) + "\n")

    polya = directory / "minus_polya.bed"
    polya.write_text("\t".join([
        CHROM, str(MINUS_POLYA - 10), str(MINUS_POLYA + 10), str(MINUS_POLYA), "120",
        "-", "three_prime_utr", "G2", "9.5", "120", "0.8", "4",
        f"{MINUS_POLYA + 16}@AATAAA", str(MINUS_POLYA)]) + "\n")
    tss = directory / "minus_tss.bed"
    tss.write_text("\t".join([
        CHROM, str(MINUS_TSS - 5), str(MINUS_TSS + 5), str(MINUS_TSS), "90", "-",
        "exon", "G2", "2.1", "90", "1.0000", "."]) + "\n")

    bams, annot = {}, ["\t".join(["read_name", "sample", "Chromosome", "Strand",
                                  "annot_gene_name", "transcript_id",
                                  "annot_transcript_name"])]
    for condition, tail_base in (("Ctrl", 60), ("Trt", 180)):
        paths = []
        for replicate in (1, 2):
            sample = f"{condition}_rep{replicate}"
            reads = []
            for tid, spans in MINUS_ISOFORMS.items():
                for index in range(6):
                    name = f"{sample}|{tid}_read{index}"
                    reads.append((name, spans, tail_base + index * 10))
                    annot.append("\t".join([name, sample, CHROM, "-", MINUS_GENE,
                                            tid, f"{MINUS_GENE}-{tid}"]))
            paths.append(_write_bam(directory / f"minus_{sample}.bam", reads,
                                    reverse=True))
        bams[condition] = paths
    read_annot = directory / "minus_read_annot.tsv"
    read_annot.write_text("\n".join(annot) + "\n")

    return dict(
        gene_name=MINUS_GENE, bam_files=bams, annotation_gtf=str(gtf),
        read_annot=str(read_annot), bed_file=str(bed),
        polya_clusters=str(polya), tss_clusters=str(tss),
        min_reads=1, verbose=False,
    )


# ------------------------------------------------------------- a fusion
# GENE5::GENEM::GENE3: three partners on two chromosomes, the middle one on the minus strand.

FUSION_NAME = "GENE5::GENEM::GENE3"
FUSION_GENES = {
    "GENE5": ("chrA", "+", [(1_000, 1_300), (3_000, 3_400)]),
    "GENEM": ("chrB", "-", [(8_000, 8_500)]),
    "GENE3": ("chrB", "+", [(2_000, 2_300), (4_000, 4_600)]),
}
#: donor>acceptor: GENE5 kept to 3,200; GENEM kept 8,400 down to 8,100 (its 5' end is the high
#: coordinate); GENE3 from the start of its exon 2.
FUSION_BREAKPOINTS = "chrA:3200>chrB:8400;chrB:8100>chrB:4000"
FUSION_ROW = (f"{FUSION_NAME}\tchrA::chrB::chrB\t{FUSION_BREAKPOINTS}\t"
              "neither;both\t2\t5\tPASS")


def _fusion_record(header, name, chrom, spans, reverse, tail, supplementary):
    read = _make_read(header, name, spans, tail)
    read.reference_id = header.references.index(chrom)
    read.is_reverse = reverse
    read.is_supplementary = supplementary
    return read


@pytest.fixture(scope="session")
def fusion_kwargs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("isotracks_fusion")
    lines = []
    for gene, (chrom, strand, spans) in FUSION_GENES.items():
        attrs = (f'gene_id "{gene}"; gene_name "{gene}"; transcript_id "{gene}-T1"; '
                 f'transcript_name "{gene}-201";')
        lines.append("\t".join([chrom, "t", "gene", str(spans[0][0]), str(spans[-1][1]),
                                ".", strand, ".", f'gene_id "{gene}"; gene_name "{gene}";']))
        lines.append("\t".join([chrom, "t", "transcript", str(spans[0][0]),
                                str(spans[-1][1]), ".", strand, ".", attrs]))
        lines += ["\t".join([chrom, "t", "exon", str(a), str(b), ".", strand, ".", attrs])
                  for a, b in spans]
    gtf = directory / "fusion.gtf"
    gtf.write_text("\n".join(lines) + "\n")
    table = directory / "fusions.tsv"
    table.write_text("# breakpoint lines\n" + FUSION_ROW + "\n")

    complete = [  # GENE5 exon 1 + half of exon 2, GENEM, GENE3 exon 2
        ("chrA", [(1_000, 1_300), (3_000, 3_200)], False),
        ("chrB", [(8_100, 8_400)], True),
        ("chrB", [(4_000, 4_600)], False),
    ]
    partial = complete[1:]
    reaching = [("chrB", [(8_100, 8_500)], True), complete[2]]
    plain = [("chrB", [(2_000, 2_300), (4_000, 4_600)], False)]
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chrA", "LN": 20_000}, {"SN": "chrB", "LN": 20_000}]}

    bams = {}
    plan = {"Ctrl": {"complete": 0, "partial": 3, "reaching": 2, "plain": 5},
            "Trt": {"complete": 4, "partial": 2, "reaching": 0, "plain": 5}}
    for condition, counts in plan.items():
        path = str(directory / f"fusion_{condition}.bam")
        records = []
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for kind, pieces in (("complete", complete), ("partial", partial),
                                 ("reaching", reaching), ("plain", plain)):
                for number in range(counts[kind]):
                    name = f"{condition}|{kind}{number}"
                    tail = 40 + 20 * number
                    for rank, (chrom, spans, reverse) in enumerate(pieces):
                        records.append(_fusion_record(
                            out.header, name, chrom, spans, reverse, tail,
                            supplementary=rank > 0))
            for record in sorted(records, key=lambda r: (r.reference_id, r.reference_start)):
                out.write(record)
        pysam.index(path)
        bams[condition] = [path]

    return dict(fusion=FUSION_NAME, fusion_table=str(table), bam_files=bams,
                reference_gtf=str(gtf), verbose=False)


@pytest.fixture(scope="session")
def fusion_tables(fusion_kwargs, tmp_path_factory):
    """IsoForge's three fusion tables for GENE5::GENEM::GENE3, in its schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path_factory.mktemp("isoforge_fusion_tables")
    prefix = str(directory / "TEST")
    spans = {g: (v[2][0][0], v[2][-1][1]) for g, v in FUSION_GENES.items()}

    calls = pa.table({
        "fusion_id": ["F000001", "F000002", "M000001"],
        "kind": ["pairwise", "pairwise", "multi_segment"],
        "filter": ["PASS", "PASS", "PASS"],
        "category": ["inter", "intra", "multi-segment"],
        "n_parts": pa.array([2, 2, 3], pa.int32()),
        "gene_ids": [["GENE5", "GENEM"], ["GENEM", "GENE3"], ["GENE5", "GENEM", "GENE3"]],
        "gene_names": [["GENE5", "GENEM"], ["GENEM", "GENE3"], ["GENE5", "GENEM", "GENE3"]],
        "reads": pa.array([4, 11, 4], pa.int32()),
        "reads_compatible": pa.array([4, 11, 11], pa.int32()),
        "constituent_ids": [[], [], ["F000001", "F000002"]],
        "reads_fusion_Ctrl": pa.array([0, 5, 0], pa.int32()),
        "reads_fusion_Trt": pa.array([4, 6, 4], pa.int32()),
    })
    pq.write_table(calls, prefix + "_isoforge_fusions.parquet")

    part_rows = [  # the multi-segment call, part by part in transcript order
        ("M000001", 1, "GENE5", "chrA", "+", [(1_000, 1_300), (3_000, 3_200)], None, 3_200, None, False),
        ("M000001", 2, "GENEM", "chrB", "-", [(8_100, 8_400)], 8_400, 8_100, False, True),
        ("M000001", 3, "GENE3", "chrB", "+", [(4_000, 4_600)], 4_000, None, True, None),
    ]
    parts = pa.table({
        "fusion_id": [r[0] for r in part_rows],
        "part": pa.array([r[1] for r in part_rows], pa.int32()),
        "gene_id": [r[2] for r in part_rows], "gene_name": [r[2] for r in part_rows],
        "chromosome": [r[3] for r in part_rows], "strand": [r[4] for r in part_rows],
        "gene_start": pa.array([spans[r[2]][0] for r in part_rows], pa.int64()),
        "gene_end": pa.array([spans[r[2]][1] for r in part_rows], pa.int64()),
        "block_starts": pa.array([[b[0] for b in r[5]] for r in part_rows], pa.list_(pa.int64())),
        "block_ends": pa.array([[b[1] for b in r[5]] for r in part_rows], pa.list_(pa.int64())),
        "junction_in": pa.array([r[6] for r in part_rows], pa.int64()),
        "junction_out": pa.array([r[7] for r in part_rows], pa.int64()),
        "boundary_in": pa.array([r[8] for r in part_rows], pa.bool_()),
        "boundary_out": pa.array([r[9] for r in part_rows], pa.bool_()),
        "chain_reads": pa.array([4, 4, 4], pa.int32()),
    })
    pq.write_table(parts, prefix + "_isoforge_fusion_parts.parquet")

    # Every piece of every supporting read, read straight from the fixture BAMs.
    loci = [("chrA", 1_000, 3_400), ("chrB", 8_000, 8_500), ("chrB", 4_000, 4_600)]
    rows = {k: [] for k in ("fusion_id", "dataset", "read_name", "spans_all", "segment",
                            "part", "chromosome", "reverse", "supplementary", "mapq",
                            "ref_start", "ref_end", "block_starts", "block_ends",
                            "query_start", "query_end", "read_length")}
    for condition, paths in fusion_kwargs["bam_files"].items():
        by_read = {}
        with pysam.AlignmentFile(paths[0]) as bam:
            for record in bam.fetch():
                if "plain" in record.query_name:
                    continue
                by_read.setdefault(record.query_name, []).append(record)
        for name, records in by_read.items():
            spans_all = "complete" in name
            for segment, record in enumerate(records, 1):
                blocks = [(s + 1, e) for s, e in record.get_blocks()]
                part = next(i + 1 for i, (c, lo, hi) in enumerate(loci)
                            if c == record.reference_name and record.reference_start < hi
                            and record.reference_end > lo
                            and (i != 1 or record.is_reverse) and (i != 2 or not record.is_reverse))
                for key, value in (("fusion_id", "M000001"), ("dataset", f"fusion_{condition}"),
                                   ("read_name", name), ("spans_all", spans_all),
                                   ("segment", segment), ("part", part),
                                   ("chromosome", record.reference_name),
                                   ("reverse", record.is_reverse),
                                   ("supplementary", record.is_supplementary), ("mapq", 60),
                                   ("ref_start", record.reference_start + 1),
                                   ("ref_end", record.reference_end),
                                   ("block_starts", [b[0] for b in blocks]),
                                   ("block_ends", [b[1] for b in blocks]),
                                   ("query_start", 0), ("query_end", 100), ("read_length", 300)):
                    rows[key].append(value)
    types = {"segment": pa.int32(), "part": pa.int32(), "mapq": pa.int32(),
             "ref_start": pa.int64(), "ref_end": pa.int64(),
             "block_starts": pa.list_(pa.int64()), "block_ends": pa.list_(pa.int64()),
             "query_start": pa.int32(), "query_end": pa.int32(), "read_length": pa.int32()}
    reads = pa.table({k: pa.array(v, types.get(k)) for k, v in rows.items()})
    pq.write_table(reads, prefix + "_isoforge_fusion_reads.parquet")
    return prefix


@pytest.fixture(scope="session")
def fusion_tables_isoforms(fusion_tables, tmp_path_factory):
    """The fusion tables with IsoForge's transcripts: the whole fusion and two of GENEM::GENE3."""
    import os
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path_factory.mktemp("isoforge_fusion_isoforms")
    prefix = str(directory / "TEST")
    for suffix in ("_isoforge_fusions.parquet", "_isoforge_fusion_parts.parquet"):
        shutil.copy(fusion_tables + suffix, prefix + suffix)
    reads = pq.read_table(fusion_tables + "_isoforge_fusion_reads.parquet").to_pandas()
    partial = sorted(set(reads.loc[~reads.spans_all & (reads.fusion_id == "M000001"), "read_name"]))
    late = set(partial[-2:])  # two reads ending at the far poly(A) site
    reads["isoform"] = [1 if s else (3 if n in late else 2)
                        for s, n in zip(reads.spans_all, reads.read_name)]
    reads["fits_isoform"] = [not (n == partial[-1]) for n in reads.read_name]
    reads["isoform_id"] = [f"ISOFORF{i:09d}" if f == "M000001" else None
                           for i, f in zip(reads.isoform, reads.fusion_id)]
    table = pa.Table.from_pandas(reads, preserve_index=False)
    table = table.set_column(table.schema.get_field_index("isoform"), "isoform",
                             pa.array(reads["isoform"], pa.int32()))
    pq.write_table(table, prefix + "_isoforge_fusion_reads.parquet")
    rows = [
        ("M000001", 1, True, 1, 3, 4, 1, "GENE5", "chrA", "+", [(1_000, 1_300), (3_000, 3_200)], None, 3_200),
        ("M000001", 1, True, 1, 3, 4, 2, "GENEM", "chrB", "-", [(8_100, 8_400)], 8_400, 8_100),
        ("M000001", 1, True, 1, 3, 4, 3, "GENE3", "chrB", "+", [(4_000, 4_600)], 4_000, None),
        ("M000001", 2, False, 2, 3, 5, 2, "GENEM", "chrB", "-", [(8_100, 8_480)], None, 8_100),
        ("M000001", 2, False, 2, 3, 5, 3, "GENE3", "chrB", "+", [(4_000, 4_450)], 4_000, None),
        ("M000001", 3, False, 2, 3, 2, 2, "GENEM", "chrB", "-", [(8_100, 8_480)], None, 8_100),
        ("M000001", 3, False, 2, 3, 2, 3, "GENE3", "chrB", "+", [(4_000, 4_600)], 4_000, None),
    ]
    isoforms = pa.table({
        "fusion_id": [r[0] for r in rows],
        "isoform": pa.array([r[1] for r in rows], pa.int32()),
        "isoform_id": [f"ISOFORF{r[1]:09d}" for r in rows],
        "complete": [r[2] for r in rows],
        "first_part": pa.array([r[3] for r in rows], pa.int32()),
        "last_part": pa.array([r[4] for r in rows], pa.int32()),
        "reads": pa.array([r[5] for r in rows], pa.int32()),
        "part": pa.array([r[6] for r in rows], pa.int32()),
        "gene_id": [r[7] for r in rows], "gene_name": [r[7] for r in rows],
        "chromosome": [r[8] for r in rows], "strand": [r[9] for r in rows],
        "block_starts": pa.array([[b[0] for b in r[10]] for r in rows], pa.list_(pa.int64())),
        "block_ends": pa.array([[b[1] for b in r[10]] for r in rows], pa.list_(pa.int64())),
        "junction_in": pa.array([r[11] for r in rows], pa.int64()),
        "junction_out": pa.array([r[12] for r in rows], pa.int64()),
    })
    pq.write_table(isoforms, prefix + "_isoforge_fusion_isoforms.parquet")
    assert os.path.exists(prefix + "_isoforge_fusion_isoforms.parquet")
    return prefix


# ------------------------------------------------ a gene and its antisense partner

#: Genes around TESTG: one on the opposite strand overlapping its 3' end, one on the
#: opposite strand far away, one on the same strand next to it.
ANTISENSE_GENES = {
    "ANTI": ("-", {"AS_LONG": [(4_500, 4_700), (6_200, 6_600), (7_400, 7_700), (9_000, 9_400)],
                   "AS_SHORT": [(8_000, 8_300), (9_000, 9_400)]}),
    "FAR": ("-", {"FAR_1": [(15_000, 15_400), (16_000, 16_300)]}),
    "NEIGH": ("+", {"NEIGH_1": [(6_900, 7_000), (7_200, 7_500)]}),
}


@pytest.fixture(scope="session")
def antisense_kwargs(tmp_path_factory):
    """TESTG with genes around it, their reads on both strands in one set of BAMs."""
    directory = tmp_path_factory.mktemp("isotracks_antisense")
    genes = {GENE: ("+", ISOFORMS), **ANTISENSE_GENES}

    lines = []
    for gene, (strand, isoforms) in genes.items():
        for tid, spans in isoforms.items():
            attrs = f'gene_id "{gene}"; gene_name "{gene}"; transcript_id "{tid}";'
            for feature, a, b in [("transcript", spans[0][0], spans[-1][1])] + [
                    ("exon", a, b) for a, b in spans]:
                lines.append("\t".join([CHROM, "test", feature, str(a), str(b), ".",
                                        strand, ".", attrs]))
    gtf = directory / "antisense.gtf"
    gtf.write_text("\n".join(lines) + "\n")

    bams, annot = {}, ["\t".join(["read_name", "sample", "Chromosome", "Strand",
                                  "annot_gene_name", "transcript_id"])]
    for condition, tail_base in (("Ctrl", 60), ("Trt", 180)):
        paths = []
        for replicate in (1, 2):
            sample = f"{condition}_rep{replicate}"
            reads = []
            for gene, (strand, isoforms) in genes.items():
                for tid, spans in isoforms.items():
                    for index in range(6 if gene == GENE else 3):
                        name = f"{sample}|{tid}_read{index}"
                        reads.append((name, spans, tail_base + index * 10, strand == "-"))
                        annot.append("\t".join([name, sample, CHROM, strand, gene, tid]))
            paths.append(_write_bam(directory / f"{sample}.bam", reads))
        bams[condition] = paths
    read_annot = directory / "read_annot.tsv"
    read_annot.write_text("\n".join(annot) + "\n")
    return dict(gene_name=GENE, bam_files=bams, annotation_gtf=str(gtf),
                read_annot=str(read_annot), min_reads=1, verbose=False)
