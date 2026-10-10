#!/usr/bin/env python3
"""Render the README's illustrations from synthetic data.

    python docs/make_examples.py

Writes ``docs/img/*.png`` from a generated gene, its reads and its modification calls.
"""

from __future__ import annotations

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isotracks import IsoTracks, theme  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")

CHROM, CHROM_LENGTH, GENE = "chr7", 44_300_000, "EXAMPLE1"
SEED = 20260920

#: Where the example gene sits on the chromosome.
LOCUS_START = 44_251_000


def _at(*spans):
    """Exon spans given as offsets from :data:`LOCUS_START`."""
    return [(LOCUS_START + s, LOCUS_START + e) for s, e in spans]


#: Three isoforms sharing a first exon, differing at the 3' end.
ISOFORMS = {
    "ENST00000111.1": _at((2_000, 2_400), (6_000, 6_300), (11_000, 13_400)),
    "ISOFORT000222":  _at((2_000, 2_400), (6_000, 6_300), (8_000, 8_200),
                          (11_000, 13_400)),
    "ISOFORT000333":  _at((2_000, 2_400), (11_000, 12_200)),
}
#: Two low-count isoforms, for the isoform filters.
MINOR = {
    "ISOFORT000444": _at((2_000, 2_400), (4_500, 4_700), (11_000, 13_400)),
    "ISOFORT000555": _at((2_000, 2_400), (9_400, 9_600), (11_000, 13_400)),
}
CODING = {tid: (LOCUS_START + a, LOCUS_START + b) for tid, (a, b) in {
    "ENST00000111.1": (2_100, 12_400),
    "ISOFORT000222":  (2_100, 12_400),
    "ISOFORT000333":  (2_100, 11_900),
    "ISOFORT000444":  (2_100, 12_400),
    "ISOFORT000555":  (2_100, 12_400),
}.items()}
#: Reads per isoform per condition, and the median tail length in each arm.
DEPTH = {"ENST00000111.1": 260, "ISOFORT000222": 90, "ISOFORT000333": 55,
         "ISOFORT000444": 8, "ISOFORT000555": 6}
TAILS = {"Ctrl": 70, "Trt": 155}
CONDITIONS = ("Ctrl", "Trt")
LABELS = {"Ctrl": "Control", "Trt": "Treated"}


# ------------------------------------------------------------------ the data

def write_gtf(path: str) -> str:
    everything = {**ISOFORMS, **MINOR}
    span = (min(s for v in everything.values() for s, _ in v),
            max(e for v in everything.values() for _, e in v))
    lines = [_gtf("gene", *span, f'gene_id "G1"; gene_name "{GENE}";')]
    for tid, spans in everything.items():
        attrs = (f'gene_id "G1"; gene_name "{GENE}"; transcript_id "{tid}"; '
                 f'transcript_name "{GENE}-{tid}";')
        lines.append(_gtf("transcript", spans[0][0], spans[-1][1], attrs))
        lines += [_gtf("exon", s, e, attrs) for s, e in spans]
    _write(path, "\n".join(lines))
    return path


def _gtf(feature, start, end, attrs):
    return "\t".join([CHROM, "ex", feature, str(start), str(end), ".", "+", ".", attrs])


def write_bed(path: str) -> str:
    rows = []
    for tid, spans in {**ISOFORMS, **MINOR}.items():
        start, end = spans[0][0] - 1, spans[-1][1]
        sizes = ",".join(str(e - s + 1) for s, e in spans) + ","
        offsets = ",".join(str(s - 1 - start) for s, _ in spans) + ","
        thick_start, thick_end = CODING[tid]
        rows.append("\t".join([
            CHROM, str(start), str(end), tid, "0", "+",
            str(thick_start - 1), str(thick_end), "0", str(len(spans)),
            sizes, offsets,
        ]))
    _write(path, "\n".join(rows))
    return path


def write_clusters(directory: str) -> tuple:
    """A TES cluster per 3' end, with its hexamer, and one TSS cluster."""
    def at(offset):
        return str(LOCUS_START + offset)

    tes = os.path.join(directory, "tes_clusters.bed")
    _write(tes, "\n".join([
        "\t".join([CHROM, at(13_390), at(13_410), at(13_400), "150", "+",
                   "three_prime_utr", "G1", "9.5", "150", "0.8", "5",
                   f"{at(13_416)}@AATAAA", at(13_400)]),
        "\t".join([CHROM, at(12_190), at(12_210), at(12_200), "70", "+",
                   "three_prime_utr", "G1", "6.1", "70", "0.5", "3",
                   f"{at(12_216)}@ATTAAA", at(12_200)]),
    ]))
    tss = os.path.join(directory, "tss_clusters.bed")
    _write(tss, "\t".join([CHROM, at(1_995), at(2_005), at(2_000), "120", "+",
                           "exon", "G1", "2.1", "120", "1.0", "."]))
    return tes, tss


def write_bams(directory: str) -> dict:
    rng = np.random.default_rng(SEED)
    files, annot = {}, []
    for condition in CONDITIONS:
        paths = []
        for replicate in (1, 2):
            reads = []
            for tid, spans in {**ISOFORMS, **MINOR}.items():
                n = DEPTH[tid] // 2
                centre = TAILS[condition] if tid == "ENST00000111.1" else 95
                tails = rng.normal(centre, centre * 0.28, n).clip(15, 400)
                for index, tail in enumerate(tails):
                    name = f"{condition}_rep{replicate}|{tid}_r{index}"
                    # Ragged 5' ends, as long reads really are.
                    trim = int(abs(rng.normal(0, 70)))
                    reads.append((name, _trim_5(spans, trim), int(tail)))
                    annot.append((name, f"{condition}_rep{replicate}", tid))
            paths.append(_write_bam(
                os.path.join(directory, f"{condition}_rep{replicate}.bam"), reads))
        files[condition] = paths

    path = os.path.join(directory, "read_annot.tsv")
    header = ["read_name", "sample", "Chromosome", "Strand", "annot_gene_name",
              "transcript_id", "cluster_supported"]
    _write(path, "\n".join(["\t".join(header)] + [
        "\t".join([name, sample, CHROM, "+", GENE, tid, "True"])
        for name, sample, tid in annot
    ]))
    return {"bams": files, "read_annot": path}


def _trim_5(spans, trim):
    """Move the 5' end in by *trim* bp, dropping exons it swallows."""
    out = []
    for start, end in spans:
        if trim <= 0:
            out.append((start, end))
        elif start + trim < end:
            out.append((start + trim, end))
            trim = 0
        else:
            trim -= end - start
    return out or [spans[-1]]


def _write_bam(path, reads):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": CHROM_LENGTH}]}
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for name, spans, tail in sorted(reads, key=lambda r: r[1][0][0]):
            out.write(_read(out.header, name, spans, tail))
    pysam.index(path)
    return path


def _read(header, name, spans, tail):
    read = pysam.AlignedSegment(header)
    read.query_name = name
    length = sum(e - s + 1 for s, e in spans)
    read.query_sequence = "A" * length
    read.query_qualities = pysam.qualitystring_to_array("I" * length)
    read.flag, read.reference_id = 0, 0
    read.reference_start = spans[0][0] - 1
    read.mapping_quality = 60
    cigar = []
    for index, (start, end) in enumerate(spans):
        if index:
            cigar.append((3, start - spans[index - 1][1] - 1))
        cigar.append((0, end - start + 1))
    read.cigartuples = cigar
    read.set_tag("pt", int(tail), value_type="i")
    return read


def write_pileups(directory: str) -> dict:
    """bedMethyl per condition: m6A sites in the last exon, weaker when treated."""
    rng = np.random.default_rng(SEED + 1)
    sites = sorted(rng.choice(
        np.arange(LOCUS_START + 11_050, LOCUS_START + 13_350), 40, replace=False))
    files = {}
    for condition in CONDITIONS:
        paths = []
        for replicate in (1, 2):
            rows = []
            for position in sites:
                coverage = int(rng.integers(30, 90))
                base = rng.uniform(0.25, 0.95)
                if condition == "Trt":
                    base *= 0.35  # the treatment removes the mark
                rows.append(_bedmethyl(int(position), "a", coverage,
                                       int(coverage * base)))
            path = os.path.join(directory, f"{condition}_rep{replicate}.bed")
            _write(path, "\n".join(rows))
            packed = path + ".gz"
            pysam.tabix_compress(path, packed, force=True)
            pysam.tabix_index(packed, preset="bed", force=True)
            paths.append(packed)
        files[condition] = paths
    return {"pileups": files, "sites": sites}


def _bedmethyl(position, code, coverage, modified):
    percent = 100.0 * modified / coverage if coverage else 0.0
    return "\t".join([
        CHROM, str(position - 1), str(position), code, str(coverage), "+",
        str(position - 1), str(position), "255,0,0", str(coverage),
        f"{percent:.2f}", str(modified), str(coverage - modified),
        "0", "0", "0", "0", "0",
    ])


def write_dmr(directory: str, sites) -> str:
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for position in sites:
        a = rng.uniform(0.35, 0.95)
        b = a * rng.uniform(0.15, 0.55)
        # A couple go the other way, so the track has both directions to show.
        if rng.random() < 0.12:
            a, b = b, a
        rows.append("\t".join([
            CHROM, str(int(position) - 1), str(int(position)), "a", "0", "+",
            "0", "60", "0", "60", "0", "0",
            f"{a:.4f}", f"{b:.4f}", "1e-9", f"{a - b:.4f}",
            "1e-9", f"{a - b:.4f}",
        ]))
    path = os.path.join(directory, "dmr_sites.bed")
    _write(path, "\n".join(rows))
    packed = path + ".gz"
    pysam.tabix_compress(path, packed, force=True)
    pysam.tabix_index(packed, preset="bed", force=True)
    return packed


def _write(path, text):
    with open(path, "w") as handle:
        handle.write(text + "\n")


# -------------------------------------------------------------- the pictures

def build(directory: str, **overrides) -> IsoTracks:
    data = build.data
    kwargs = dict(
        gene_name=GENE,
        bam_files=data["bams"],
        annotation_gtf=data["gtf"],
        read_annot=data["read_annot"],
        bed_file=data["bed"],
        polya_clusters=data["tes"],
        tss_clusters=data["tss"],
        min_reads_per_condition=20,
        condition_labels=LABELS,
        verbose=False,
    )
    kwargs.update(overrides)
    return IsoTracks(**kwargs)


def save(view: IsoTracks, name: str, **plot_kwargs) -> None:
    view.plot(output=os.path.join(IMG, name), formats=("png",), dpi=220,
              **plot_kwargs)
    plt.close("all")
    print(f"  {name}.png")


def main() -> int:
    os.makedirs(IMG, exist_ok=True)
    directory = tempfile.mkdtemp(prefix="isotracks_examples_")
    print(f"generating synthetic data in {directory}")

    data = write_bams(directory)
    data["gtf"] = write_gtf(os.path.join(directory, "models.gtf"))
    data["bed"] = write_bed(os.path.join(directory, "models.bed"))
    data["tes"], data["tss"] = write_clusters(directory)
    pileups = write_pileups(directory)
    data["pileups"] = pileups["pileups"]
    data["dmr"] = write_dmr(directory, pileups["sites"])
    build.data = data

    theme.use(width="single", readability="comfortable")
    print("rendering:")

    save(build(directory), "anatomy")

    save(build(directory), "sort_waterfall", sort_style="waterfall",
         region=(CHROM, LOCUS_START + 10_800, LOCUS_START + 14_200, "+"))
    save(build(directory), "sort_diamond", sort_style="diamond",
         region=(CHROM, LOCUS_START + 10_800, LOCUS_START + 14_200, "+"))

    save(build(directory, modkit_pileup=data["pileups"], mod_codes=("a",),
               min_mod_fraction=0.2),
         "modifications", region=(CHROM, LOCUS_START + 10_800, LOCUS_START + 13_800, "+"))
    save(build(directory, modkit_pileup=data["pileups"], mod_codes=("a",),
               min_mod_fraction=0.2,
               mod_diff={"Control vs Treated": data["dmr"]}),
         "modification_diff", region=(CHROM, LOCUS_START + 10_800, LOCUS_START + 13_800, "+"))

    save(build(directory, modkit_pileup=data["pileups"], mod_codes=("a",),
               min_mod_fraction=0.2, compact=False),
         "roomy")

    save(build(directory), "highlight",
         highlight=["junction_2", "3_utr:ENST00000111.1"])

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
