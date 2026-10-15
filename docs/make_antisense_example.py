#!/usr/bin/env python3
"""Render the README's opposite-strand figure from an IsoForge run.

    python docs/make_antisense_example.py --gene COPS6 --isoforge results/project \\
        --bam "Control:c1.bam,c2.bam;Treated:t1.bam,t2.bam"

Writes ``docs/img/antisense.png``: the gene, and below it the genes on the opposite strand, drawn whole.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gene", default="COPS6")
    parser.add_argument("--isoforge", required=True, metavar="PREFIX")
    parser.add_argument("--bam", type=parse_bam_spec, metavar="SPEC")
    parser.add_argument("--max-isoforms", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(IMG, exist_ok=True)
    theme.use(width="single", readability="comfortable")
    view = IsoTracks(gene_name=args.gene, isoforge=args.isoforge, bam_files=args.bam,
                     max_isoforms=args.max_isoforms, antisense="whole", verbose=False)
    view.plot(output=os.path.join(IMG, "antisense"), formats=("png",), dpi=220)
    plt.close("all")
    print("  antisense.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
