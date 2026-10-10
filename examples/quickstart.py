#!/usr/bin/env python3
"""A worked example: one gene from an IsoForge run and its BAMs.

    python examples/quickstart.py SAT1

Edit the paths below, then run it.  Writes the whole-gene figure, a close-up of
the 3' end in both read orderings, and the poly(A) tail lengths as a table.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from isotracks import IsoTracks, theme

# ---------------------------------------------------------------- your paths

ISOFORGE = "results/project"          # the --o prefix given to isoforge annotate
BAM_DIR = "alignments"
OUTPUT_DIR = "figures"

BAM_FILES = {
    "Control": [f"{BAM_DIR}/control_rep{i}.bam" for i in (1, 2, 3)],
    "Treated": [f"{BAM_DIR}/treated_rep{i}.bam" for i in (1, 2, 3)],
}

# Optional: BED12 models with coding extent, so CDS and UTRs are drawn apart.
BED_FILE = None

MIN_READS_PER_CONDITION = 5


def main(gene: str) -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    theme.use(width="single", readability="comfortable")

    view = IsoTracks(
        gene_name=gene,
        isoforge=ISOFORGE,
        bam_files=BAM_FILES,
        bed_file=BED_FILE,
        min_reads_per_condition=MIN_READS_PER_CONDITION,
    )
    if not view.transcripts():
        print("no isoform passed the read threshold; lower MIN_READS_PER_CONDITION")
        return 1

    view.plot(output=f"{OUTPUT_DIR}/{gene}")

    last_exon = view.zoom("last_exon", padding=400)
    view.plot(region=last_exon, output=f"{OUTPUT_DIR}/{gene}_3prime")
    view.plot(region=last_exon, sort_style="diamond",
              output=f"{OUTPUT_DIR}/{gene}_3prime_diamond")

    table = view.polya_table()
    table.to_csv(f"{OUTPUT_DIR}/{gene}_polya.csv", index=False)
    print(table.groupby("condition")["polya_length"].agg(["count", "median"]).to_string())

    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "SAT1"))
