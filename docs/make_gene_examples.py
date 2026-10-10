#!/usr/bin/env python3
"""Render the README's close-up and shortened-intron figures from an IsoForge run.

    python docs/make_gene_examples.py --gene COPZ1 --isoforge results/project \\
        --bam "Control:c1.bam,c2.bam;Treated:t1.bam,t2.bam" --bed models.bed

Writes ``docs/img/<gene>_{default,detail,scaled,fixed}.png``.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isotracks import IsoTracks, theme  # noqa: E402
from isotracks.cli import parse_bam_spec  # noqa: E402

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")

#: File stem and the plot() options of each figure.
FIGURES = [
    ("default", {}),
    ("detail", {"detail": "last_exon"}),
    ("scaled", {"introns": "scaled"}),
    ("fixed", {"introns": "fixed"}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gene", default="COPZ1")
    parser.add_argument("--isoforge", required=True, metavar="PREFIX")
    parser.add_argument("--bam", required=True, type=parse_bam_spec, metavar="SPEC")
    parser.add_argument("--annotation-gtf", metavar="GTF")
    parser.add_argument("--bed", metavar="BED12")
    parser.add_argument("--reference-gtf", metavar="GTF")
    parser.add_argument("--max-isoforms", type=int, default=2)
    parser.add_argument("--min-reads-per-condition", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(IMG, exist_ok=True)
    theme.use(width="single", readability="comfortable")
    for name, options in FIGURES:
        view = IsoTracks(
            gene_name=args.gene, isoforge=args.isoforge, bam_files=args.bam,
            annotation_gtf=args.annotation_gtf, bed_file=args.bed,
            reference_gtf=args.reference_gtf, max_isoforms=args.max_isoforms,
            min_reads_per_condition=args.min_reads_per_condition, verbose=False,
        )
        stem = f"{args.gene.lower()}_{name}"
        view.plot(output=os.path.join(IMG, stem), formats=("png",), dpi=220, **options)
        plt.close("all")
        print(f"  {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
