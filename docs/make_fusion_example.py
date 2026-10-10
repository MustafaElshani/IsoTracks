#!/usr/bin/env python3
"""Render the README's fusion figures from an IsoForge run.

    python docs/make_fusion_example.py --fusion RIMS2::ATP6V1C1 --isoforge results/project \\
        --bam "Control:c1.bam,c2.bam;Treated:t1.bam,t2.bam" --reference-gtf gencode.gtf

Writes ``docs/img/fusion_all.png`` and ``docs/img/fusion_schematic.png``, the schematic alone.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.transforms import Bbox  # noqa: E402

from isotracks import FusionView, theme  # noqa: E402
from isotracks.cli import parse_bam_spec  # noqa: E402

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusion", default="RIMS2::ATP6V1C1")
    parser.add_argument("--isoforge", required=True, metavar="PREFIX")
    parser.add_argument("--bam", type=parse_bam_spec, metavar="SPEC")
    parser.add_argument("--reference-gtf", metavar="GTF")
    args = parser.parse_args()

    os.makedirs(IMG, exist_ok=True)
    theme.use(width="single", readability="comfortable")
    view = FusionView(args.fusion, isoforge=args.isoforge, bam_files=args.bam,
                      reference_gtf=args.reference_gtf, verbose=False)
    figure = view.plot(output=os.path.join(IMG, "fusion_all"), formats=("png",), dpi=220)
    print("  fusion_all.png")
    save_schematic(view, figure, os.path.join(IMG, "fusion_schematic.png"))
    print("  fusion_schematic.png")
    plt.close("all")
    return 0


def save_schematic(view, figure, path: str) -> None:
    """Save the schematic's row of the figure alone."""
    track = view.canvas.of_type("FusionSchematicTrack")[0]
    renderer = figure.canvas.get_renderer()
    ink = Bbox.union([track.ax.get_tightbbox(renderer),
                      track.gutter_ax.get_tightbbox(renderer)])
    pad = 0.04 * figure.dpi
    box = Bbox.from_extents(ink.x0 - pad, track.ax.get_window_extent().y0,
                            ink.x1 + pad, ink.y1 + pad)
    figure.savefig(path, dpi=220, bbox_inches=box.transformed(figure.dpi_scale_trans.inverted()),
                   facecolor="white")


if __name__ == "__main__":
    raise SystemExit(main())
