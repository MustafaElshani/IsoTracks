"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .highlight import HIGHLIGHT_STYLES
from .introns import INTRON_GENOMIC, INTRON_MODES
from .browser import (
    COVERAGE_DRAWN,
    DEFAULT_MAX_READS,
    COVERAGE_READS,
    MOD_REGIONS,
    MOD_REGIONS_EXONS,
)

EPILOG = """\
examples:
  # one gene from an IsoForge run and its BAMs
  isotracks plot --gene SAT1 --isoforge results/project \\
      --bam "Control:c1.bam,c2.bam;Treated:t1.bam,t2.bam" \\
      --output figures/SAT1

  # close-up of the 3' end
  isotracks plot --gene SAT1 --isoforge results/project \\
      --bam "Control:c1.bam;Treated:t1.bam" --zoom last_exon --output SAT1_3prime.pdf

  # a gene list, one figure each
  isotracks batch --genes genes.txt --outdir figures --format pdf \\
      --isoforge results/project --bam "Control:c1.bam;Treated:t1.bam"

  # a fusion
  isotracks plot --fusion RIMS2::ATP6V1C1 --isoforge results/project \\
      --bam "Control:c1.bam;Treated:t1.bam" --reference-gtf gencode.gtf \\
      --output figures/RIMS2_ATP6V1C1

  # direct RNA modifications: m6A and pseudouridine in separate tracks
  isotracks plot --gene ACTB --isoforge results/project \\
      --bam "Control:c1.bam;Treated:t1.bam" \\
      --modkit "Control:c1.bed.gz;Treated:t1.bed.gz" --mods a 17802 \\
      --output figures/ACTB

  # poly(A) tail lengths as a table
  isotracks polya --gene SAT1 --isoforge results/project \\
      --bam "Control:c1.bam;Treated:t1.bam" -o SAT1_polya.csv

  # gene, isoform and exon coordinates
  isotracks info --gene SAT1 --isoforge results/project
"""


def parse_groups(text: str, what: str = "BAM files") -> Dict[str, List[str]]:
    """``"Ctrl:a,b;Trt:c"`` to ``{"Ctrl": ["a", "b"], "Trt": ["c"]}``."""
    groups: Dict[str, List[str]] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"{part!r} is not 'Condition:member'; separate conditions with ';' "
                "and members with ','"
            )
        condition, _, members = part.partition(":")
        condition = condition.strip()
        names = [m.strip() for m in members.split(",") if m.strip()]
        if not condition or not names:
            raise argparse.ArgumentTypeError(f"{part!r} names no condition or no {what}")
        groups.setdefault(condition, []).extend(names)
    if not groups:
        raise argparse.ArgumentTypeError(f"no {what} given")
    return groups


def parse_bam_spec(text: str) -> Dict[str, List[str]]:
    """``"Ctrl:a.bam,b.bam;Trt:c.bam"`` to ``{"Ctrl": [...], "Trt": [...]}``."""
    return parse_groups(text, "BAM files")


def parse_dataset_spec(text: str) -> Dict[str, List[str]]:
    """``"Control:rep1,rep2;Treated:rep3"`` to the fusion tables' datasets per condition."""
    return parse_groups(text, "datasets")


def parse_width(text: str):
    """``"single"`` to itself and ``"176"`` to 176.0 millimetres."""
    from .theme import COLUMN_WIDTHS

    if text in COLUMN_WIDTHS:
        return text
    try:
        width = float(text)
    except ValueError:
        width = 0.0
    if width <= 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a width; use millimetres or one of {', '.join(COLUMN_WIDTHS)}")
    return width


def parse_region(text: str) -> Tuple[str, int, int, Optional[str]]:
    """``"chr1:1,000,000-1,050,000:+"`` to ``("chr1", 1000000, 1050000, "+")``."""
    parts = text.split(":")
    if len(parts) < 2 or "-" not in parts[1]:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not 'chr:start-end[:strand]'"
        )
    chrom = parts[0]
    start_text, _, end_text = parts[1].partition("-")
    strand = parts[2] if len(parts) > 2 else None
    if strand not in (None, "+", "-"):
        raise argparse.ArgumentTypeError(f"strand must be '+' or '-', got {strand!r}")
    try:
        start = int(start_text.replace(",", ""))
        end = int(end_text.replace(",", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"non-numeric coordinates in {text!r}") from None
    return (chrom, start, end, strand)


def parse_diff_tables(values: Optional[Sequence[str]]) -> Optional[Dict[str, str]]:
    """``--dmr`` into ``{label: path}``."""
    if not values:
        return None
    tables = {}
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                f"{value!r} is not 'LABEL:path'; e.g. \"Control vs Treated:sites.bed.gz\""
            )
        label, _, path = value.rpartition(":")
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise argparse.ArgumentTypeError(f"{value!r} names no label or no file")
        tables[label] = path
    return tables


def parse_mod_groups(values: Optional[Sequence[str]]) -> Optional[List[List[str]]]:
    """``--mods`` into modification groups, one track per group.

        --mods a,17802        one track with m6A and pseudouridine
        --mods a 17802        a track each
        --mods a,17802 m      m6A and pseudouridine together, then 5mC
    """
    if not values:
        return None
    groups = []
    for value in values:
        codes = [code.strip() for code in value.split(",") if code.strip()]
        if not codes:
            raise argparse.ArgumentTypeError(f"{value!r} names no modification codes")
        groups.append(codes)
    return groups


def parse_mapping(text: Optional[str]) -> Optional[Dict[str, str]]:
    """``"Ctrl:Control,Trt:Treated"`` to a dict."""
    if not text:
        return None
    out: Dict[str, str] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise argparse.ArgumentTypeError(f"{pair!r} is not 'key:value'")
        key, _, value = pair.partition(":")
        out[key.strip()] = value.strip()
    return out


def read_gene_list(path: str) -> List[str]:
    """Gene symbols, one per line, from a file or ``-`` for stdin."""
    handle = sys.stdin if path == "-" else open(path)
    try:
        genes = [line.strip() for line in handle]
    finally:
        if handle is not sys.stdin:
            handle.close()
    return [g for g in genes if g and not g.startswith("#")]


# --------------------------------------------------------------- arguments

def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("input")
    data.add_argument(
        "--isoforge", metavar="PREFIX",
        help="An IsoForge run: its output prefix, its directory, or one of its files. "
             "Fills --annotation-gtf, --read-annot, --polya, --tss, --fusion-table, "
             "--bed (<prefix>_isoforge_sqanti3.bed) and --bam (<prefix>_alignments/); "
             "any of those given explicitly takes precedence.")
    data.add_argument(
        "--bam", "-b", type=parse_bam_spec, metavar="SPEC",
        help='Sorted, indexed BAMs per condition: "Control:c1.bam,c2.bam;Treated:t1.bam". '
             "Replicates within a condition are pooled. Not needed when the IsoForge run "
             "kept its alignments (isoforge annotate --write_bam).")

    annot = parser.add_argument_group("annotation")
    annot.add_argument("--reference-gtf", metavar="GTF",
                       help="Reference annotation, e.g. GENCODE.")
    annot.add_argument("--annotation-gtf", metavar="GTF",
                       help="The run's transcript models (<prefix>_isoforge.gtf).")
    annot.add_argument("--read-annot", metavar="PARQUET|TSV",
                       help="Read-to-transcript table (<prefix>_isoforge_read_annot.parquet). "
                            "Without it all reads form one stack per condition.")
    annot.add_argument("--read-filter", default="supported", choices=["supported", "all"],
                       help="'supported' keeps the whole reads of the transcripts in "
                            "--annotation-gtf; 'all' keeps every read.")
    annot.add_argument("--max-frac-a", type=float, default=None, metavar="F",
                       help="Also drop reads with no poly(A) signal and an A fraction above F.")
    annot.add_argument("--bed", metavar="BED12",
                       help="BED12 isoform models with coding extent. Without it models are "
                            "drawn from the GTF exons.")
    annot.add_argument("--polya", metavar="BED",
                       help="Clustered 3' end sites (tes_clusters.bed).")
    annot.add_argument("--tss", metavar="BED",
                       help="Clustered transcription start sites (tss_clusters.bed).")

    mods = parser.add_argument_group("RNA modifications (direct RNA)")
    mods.add_argument("--modkit", metavar="SPEC",
                      help='modkit pileup bedMethyl per condition, bgzipped and tabix-indexed: '
                           '"Control:c1.bed.gz;Treated:t1.bed.gz".')
    mods.add_argument("--mods", nargs="+", metavar="CODES",
                      help="Modification codes to draw. Codes joined by commas share a track; "
                           "separate arguments get a track each. Default: a,17802.")
    mods.add_argument("--dmr", nargs="+", metavar="LABEL:FILE",
                      help='modkit dmr pair site tables, bgzipped and tabix-indexed: '
                           '"Control vs Treated:sites.bed.gz".')
    mods.add_argument("--max-diff-pvalue", type=float, default=0.05, metavar="P",
                      help="Largest p-value of a differential site that is drawn.")
    mods.add_argument("--min-diff-effect", type=float, default=0.2, metavar="F",
                      help="Smallest change in modified fraction that is drawn.")
    mods.add_argument("--balanced-diff", action="store_true",
                      help="Use modkit's replicate-balanced effect size and p-value.")
    mods.add_argument("--min-mod-coverage", type=int, default=10, metavar="N",
                      help="Least reads at a site for it to be drawn.")
    mods.add_argument("--min-mod-fraction", type=float, default=0.3, metavar="F",
                      help="Least modified fraction at a site for it to be drawn.")
    mods.add_argument("--mod-regions", choices=MOD_REGIONS, default=MOD_REGIONS_EXONS,
                      help="Draw modification sites in the exons of the drawn isoforms, or "
                           "anywhere in the window.")

    cov = parser.add_argument_group("coverage")
    cov.add_argument("--coverage-reads", choices=COVERAGE_READS, default=COVERAGE_DRAWN,
                     help="Build coverage from the reads of the drawn isoforms, or from every "
                          "read of the gene.")

    select = parser.add_argument_group("isoform selection")
    select.add_argument("--min-reads", type=int, default=5, metavar="N",
                        help="Least reads per isoform, summed over conditions.")
    select.add_argument("--min-reads-per-condition", type=int, default=None, metavar="N",
                        help="Least reads per isoform in every condition.")
    select.add_argument("--max-isoforms", type=int, default=12, metavar="N",
                        help="Most isoforms drawn, best supported first; 0 for no limit.")
    select.add_argument("--antisense", nargs="?", const="window", choices=("window", "whole"),
                        help="Also draw the genes on the opposite strand in the window, with "
                             "their isoforms, reads and coverage. 'whole' widens the window "
                             "to draw them whole.")


def add_style_arguments(parser: argparse.ArgumentParser) -> None:
    style = parser.add_argument_group("appearance")
    style.add_argument(
        "--highlight", nargs="+", metavar="REGION",
        help="Regions to mark with a band across every track: exon_<N>, first_exon, "
             "last_exon, intron_<N>, junction_<N>, 5_utr, 3_utr, cds, or a range "
             "(chr22:31,950,000-31,951,000). Append ':TRANSCRIPT' to measure on one isoform.")
    style.add_argument("--highlight-style", choices=HIGHLIGHT_STYLES, default="both",
                       help="A faint box with a dotted edge, the box alone, or the edge alone.")
    style.add_argument(
        "--detail", metavar="REGION",
        help="Region to draw as a close-up beside the figure: last_exon, exon_<N>, 3_utr, "
             "junction_<N>, 3_prime, 5_prime or a range.")
    style.add_argument("--detail-share", type=float, default=0.4, metavar="F",
                       help="Fraction of the data width the close-up takes.")
    style.add_argument("--detail-padding", type=int, default=200, metavar="BP",
                       help="Base pairs either side of the close-up region.")
    style.add_argument(
        "--introns", choices=INTRON_MODES, default=None,
        help="Draw introns at true length ('genomic'), shrunk by one factor ('scaled') or "
             "all at one width ('fixed'). Default: genomic for a gene, scaled for a fusion.")
    style.add_argument("--intron-share", type=float, default=0.3, metavar="F",
                       help="With --introns scaled, the fraction of the width introns take.")
    style.add_argument("--roomy", dest="compact", action="store_false",
                       help="Use the taller track heights.")
    style.add_argument("--no-gene-model", dest="gene_model", action="store_false",
                       help="Leave out the gene row.")
    style.add_argument("--width", default="single", type=parse_width, metavar="W",
                       help="Printed width: single, onehalf, double, slide, or millimetres.")
    style.add_argument("--readability", default="comfortable",
                       choices=["strict", "comfortable", "large"], help="Type size tier.")
    style.add_argument("--sort-style", default="waterfall", choices=["waterfall", "diamond"],
                       help="Order of reads within a stack.")
    style.add_argument("--max-reads", type=int, default=DEFAULT_MAX_READS, metavar="N",
                       help="Most reads drawn per stack; 0 draws all.")
    style.add_argument("--polya-scale", type=float, default=1.0, metavar="X",
                       help="Base pairs drawn per nucleotide of poly(A) tail.")
    style.add_argument("--padding", default="auto", metavar="BP",
                       help="Base pairs of room at the 3' end, or 'auto' to fit the longest tail.")
    style.add_argument("--condition-labels", type=parse_mapping, metavar="MAP",
                       help='Display names, e.g. "Ctrl:Control,Trt:Treated".')
    style.add_argument("--condition-colors", type=parse_mapping, metavar="MAP",
                       help='Hex colours, e.g. "Control:#9AA5B1,Treated:#1B2A49".')
    style.add_argument("--formats", nargs="+", metavar="EXT",
                       help="Formats to write, e.g. pdf png. Default: the extension of "
                            "--output, or pdf and png when it has none.")
    style.add_argument("--dpi", type=int, default=600, help="Resolution of raster output.")
    style.add_argument("--no-shared-coverage-scale", dest="share_coverage_scale",
                       action="store_false",
                       help="Scale each condition's coverage independently.")


def view_kwargs(args) -> dict:
    groups = parse_mod_groups(getattr(args, "mods", None))
    extra = {"mod_codes": groups} if groups else {}
    return dict(
        **extra,
        bam_files=args.bam,
        isoforge=getattr(args, "isoforge", None),
        reference_gtf=args.reference_gtf,
        annotation_gtf=args.annotation_gtf,
        read_annot=args.read_annot,
        read_filter=getattr(args, "read_filter", "supported"),
        max_frac_a=getattr(args, "max_frac_a", None),
        bed_file=getattr(args, "bed", None),
        polya_clusters=getattr(args, "polya", None),
        tss_clusters=getattr(args, "tss", None),
        modkit_pileup=parse_bam_spec(args.modkit) if getattr(args, "modkit", None) else None,
        min_mod_coverage=getattr(args, "min_mod_coverage", 10),
        min_mod_fraction=getattr(args, "min_mod_fraction", 0.3),
        mod_diff=parse_diff_tables(getattr(args, "dmr", None)),
        max_diff_p_value=getattr(args, "max_diff_pvalue", 0.05),
        min_diff_effect=getattr(args, "min_diff_effect", 0.2),
        balanced_diff=getattr(args, "balanced_diff", False),
        mod_regions=getattr(args, "mod_regions", MOD_REGIONS_EXONS),
        coverage_reads=getattr(args, "coverage_reads", COVERAGE_DRAWN),
        antisense=getattr(args, "antisense", None),
        gene_model=getattr(args, "gene_model", True),
        compact=getattr(args, "compact", False),
        min_reads=args.min_reads,
        min_reads_per_condition=args.min_reads_per_condition,
        max_isoforms=args.max_isoforms or None,
        condition_labels=getattr(args, "condition_labels", None),
        condition_colors=getattr(args, "condition_colors", None),
        verbose=not getattr(args, "quiet", False),
    )


def plot_kwargs(args) -> dict:
    padding = args.padding
    if padding != "auto":
        try:
            padding = int(padding)
        except ValueError:
            raise SystemExit(f"--padding must be an integer or 'auto', got {padding!r}")
    return dict(
        padding=padding,
        formats=getattr(args, "formats", None),
        width=args.width,
        sort_style=args.sort_style,
        max_reads=args.max_reads or None,
        polya_scale=args.polya_scale,
        share_coverage_scale=args.share_coverage_scale,
        highlight=getattr(args, "highlight", None) or None,
        highlight_style=getattr(args, "highlight_style", "both"),
        detail=getattr(args, "detail", None),
        detail_share=getattr(args, "detail_share", 0.4),
        detail_padding=getattr(args, "detail_padding", 200),
        introns=getattr(args, "introns", None) or INTRON_GENOMIC,
        intron_share=getattr(args, "intron_share", 0.3),
        dpi=args.dpi,
    )


# ---------------------------------------------------------------- commands

def require_bams(args) -> None:
    """Stop unless BAMs were given or the IsoForge run kept its alignments."""
    from .isoforge import find_outputs

    isoforge = getattr(args, "isoforge", None)
    if args.bam or (isoforge and find_outputs(isoforge).get("bam_files")):
        return
    raise SystemExit("no BAM files: give --bam, or an --isoforge run that kept its "
                     "alignments (isoforge annotate --write_bam)")


def cmd_plot(args) -> int:
    from . import theme as theme_module
    from .browser import IsoTracks

    theme_module.use(width=args.width, readability=args.readability)
    if getattr(args, "fusion", None):
        return cmd_fusion(args)
    require_bams(args)

    region = args.region if args.region else None
    view = IsoTracks(gene_name=args.gene, region=region, **view_kwargs(args))

    window = view.zoom(args.zoom, padding=args.zoom_padding) if args.zoom else None
    view.plot(output=args.output, region=window, **plot_kwargs(args))
    return 0


def cmd_fusion(args) -> int:
    """``plot --fusion NAME``: the fusion figure instead of a gene's."""
    from .fusion_view import FusionView

    isoforge = getattr(args, "isoforge", None)
    if not args.fusion_table and not isoforge:
        raise SystemExit("--fusion needs --isoforge or --fusion-table, the IsoForge tables it is in")
    view = FusionView(
        args.fusion, bam_files=args.bam, fusion_table=args.fusion_table,
        isoforge=isoforge, conditions=args.conditions,
        reference_gtf=args.reference_gtf, annotation_gtf=args.annotation_gtf,
        reads=args.fusion_reads, introns=args.introns or "scaled",
        condition_labels=getattr(args, "condition_labels", None),
        compact=getattr(args, "compact", True),
        schematic=getattr(args, "fusion_schematic", True),
        tss=getattr(args, "tss", None), polya=getattr(args, "polya", None),
        verbose=not args.quiet,
    )
    view.plot(output=args.output, formats=getattr(args, "formats", None), dpi=args.dpi)
    return 0


def cmd_batch(args) -> int:
    import matplotlib.pyplot as plt

    from . import theme as theme_module
    from .browser import IsoTracks

    theme_module.use(width=args.width, readability=args.readability)
    require_bams(args)
    genes = read_gene_list(args.genes)
    if not genes:
        raise SystemExit(f"no gene names in {args.genes}")

    os.makedirs(args.outdir, exist_ok=True)
    shared = view_kwargs(args)
    drawing = plot_kwargs(args)

    failures: List[Tuple[str, str]] = []
    for index, gene in enumerate(genes, 1):
        output = os.path.join(args.outdir, f"{gene}.{args.format}")
        if args.skip_existing and os.path.exists(output):
            print(f"[{index}/{len(genes)}] {gene}: exists, skipping")
            continue
        print(f"[{index}/{len(genes)}] {gene}")
        try:
            view = IsoTracks(gene_name=gene, **shared)
            window = view.zoom(args.zoom, padding=args.zoom_padding) if args.zoom else None
            view.plot(output=output, region=window, **drawing)
        except Exception as error:  # one bad gene must not end the run
            failures.append((gene, str(error)))
            print(f"    failed: {error}", file=sys.stderr)
        finally:
            plt.close("all")

    done = len(genes) - len(failures)
    print(f"\n{done}/{len(genes)} figures written to {args.outdir}")
    if failures:
        print("failed:", file=sys.stderr)
        for gene, error in failures:
            print(f"  {gene}: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_index(args) -> int:
    from .annotation import ReadAnnotation
    from . import cache as cache_module

    annotation = ReadAnnotation(args.read_annot)
    if args.drop:
        removed = cache_module.remove(args.read_annot)
        print("index removed" if removed else "no index to remove")
        return 0

    existing = annotation.cache_manifest
    if existing is not None and not args.force:
        print(f"index is already current ({existing.rows:,} rows); use --force to rebuild")
        return 0

    size_gb = os.path.getsize(args.read_annot) / 1e9
    print(f"indexing {args.read_annot} ({size_gb:.1f} GB)")

    def report(rows: int) -> None:
        print(f"\r  {rows:,} rows", end="", flush=True)

    manifest = annotation.build_cache(buckets=args.buckets, progress=report)
    print(f"\r  {manifest.rows:,} rows indexed into {manifest.buckets} buckets")
    print(f"index written to {cache_module.cache_path_for(args.read_annot)}")
    return 0


def cmd_polya(args) -> int:
    from .browser import IsoTracks

    require_bams(args)
    view = IsoTracks(gene_name=args.gene, region=args.region, **view_kwargs(args))
    table = view.polya_table()
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"wrote {len(table):,} reads to {args.output}")
    else:
        table.to_csv(sys.stdout, index=False)
    return 0


def cmd_info(args) -> int:
    """Gene, isoform and exon coordinates, as a report or as a table."""
    from .browser import IsoTracks

    view = IsoTracks(gene_name=args.gene, region=args.region, **view_kwargs(args))
    every = not args.drawn_only

    if args.format == "text":
        text = view.describe(transcript_id=args.transcript,
                             exons=not args.no_exons, all_isoforms=every)
        if args.output:
            with open(args.output, "w") as handle:
                handle.write(text + "\n")
            print(f"wrote {args.output}")
        else:
            print(text)
        return 0

    table = (view.exons(args.transcript, all_isoforms=every) if args.exons_table
             else view.isoform_table(args.transcript, all_isoforms=every))
    separator = "\t" if args.format == "tsv" else ","
    if args.output:
        table.to_csv(args.output, index=False, sep=separator)
        print(f"wrote {len(table):,} row(s) to {args.output}")
    else:
        table.to_csv(sys.stdout, index=False, sep=separator)
    return 0


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show a default only where an option has one."""

    def _get_help_string(self, action):
        if action.default in (None, True, False, argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isotracks",
        description="Isoform, read and poly(A) figures from IsoForge long-read RNA-seq results.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"isotracks {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    plot = subparsers.add_parser(
        "plot", help="draw one gene or fusion",
        description="Draw one gene, region or fusion as a multi-track figure.",
        formatter_class=_HelpFormatter,
    )
    target = plot.add_mutually_exclusive_group(required=True)
    target.add_argument("--gene", "-g", help="Gene symbol.")
    target.add_argument("--region", "-r", type=parse_region,
                        help="Genomic window as chr:start-end[:strand].")
    target.add_argument("--fusion", metavar="NAME",
                        help="A fusion by its genes (RIMS2::ATP6V1C1) or call ID (F000017).")
    add_input_arguments(plot)
    fusion = plot.add_argument_group("fusion")
    fusion.add_argument("--fusion-table", metavar="PREFIX",
                        help="IsoForge's fusion tables: their prefix, their directory or one "
                             "of the *_isoforge_fusion*.parquet files. Or a file of "
                             "breakpoint lines, with --bam and a GTF.")
    fusion.add_argument("--fusion-reads", choices=("all", "complete"), default="all",
                        help="Draw every transcript of the call, or only those through every "
                             "partner.")
    fusion.add_argument("--no-fusion-schematic", dest="fusion_schematic",
                        action="store_false", help="Leave out the schematic above the panels.")
    fusion.add_argument("--conditions", type=parse_dataset_spec, metavar="SPEC",
                        help='The fusion tables\' datasets per condition, without --bam: '
                             '"Control:rep1,rep2;Treated:rep3".')
    add_style_arguments(plot)
    plot.add_argument("--zoom", metavar="TARGET",
                      help="Draw only first_exon, last_exon, exon_<N>, 5_prime, 3_prime or "
                           "whole_gene.")
    plot.add_argument("--zoom-padding", type=int, default=200, metavar="BP",
                      help="Base pairs either side of the zoom target.")
    plot.add_argument("--output", "-o", required=True, metavar="FILE",
                      help="Output file, or a stem with no extension to write every format "
                           "in --formats.")
    plot.add_argument("--quiet", "-q", action="store_true", help="Print no progress.")
    plot.set_defaults(func=cmd_plot)

    batch = subparsers.add_parser(
        "batch", help="draw many genes",
        description="Draw one figure per gene from a list.",
        formatter_class=_HelpFormatter,
    )
    batch.add_argument("--genes", required=True, metavar="FILE",
                       help="Gene symbols, one per line; '-' reads stdin.")
    add_input_arguments(batch)
    add_style_arguments(batch)
    batch.add_argument("--zoom", metavar="TARGET", help="Zoom target applied to every gene.")
    batch.add_argument("--zoom-padding", type=int, default=200, metavar="BP",
                       help="Base pairs either side of the zoom target.")
    batch.add_argument("--outdir", "-o", required=True, metavar="DIR",
                       help="Directory for the figures.")
    batch.add_argument("--format", default="pdf", metavar="EXT", help="Output format.")
    batch.add_argument("--skip-existing", action="store_true",
                       help="Skip genes that already have a figure.")
    batch.add_argument("--quiet", "-q", action="store_true", help="Print no progress.")
    batch.set_defaults(func=cmd_batch)

    polya = subparsers.add_parser(
        "polya", help="export poly(A) tail lengths",
        description="Write per-read poly(A) tail lengths as CSV.",
        formatter_class=_HelpFormatter,
    )
    polya_target = polya.add_mutually_exclusive_group(required=True)
    polya_target.add_argument("--gene", "-g", help="Gene symbol.")
    polya_target.add_argument("--region", "-r", type=parse_region,
                              help="Genomic window as chr:start-end[:strand].")
    add_input_arguments(polya)
    polya.add_argument("--output", "-o", metavar="CSV",
                       help="Output CSV; stdout when omitted.")
    polya.add_argument("--quiet", "-q", action="store_true", help="Print no progress.")
    polya.set_defaults(func=cmd_polya)

    info = subparsers.add_parser(
        "info", help="gene, isoform and exon coordinates",
        description="Report a gene's isoforms: exon count, transcript length, genomic span, "
                    "exon coordinates and supporting reads. Coordinates are 1-based and "
                    "inclusive.",
        formatter_class=_HelpFormatter,
    )
    info_target = info.add_mutually_exclusive_group(required=True)
    info_target.add_argument("--gene", "-g", help="Gene symbol.")
    info_target.add_argument("--region", "-r", type=parse_region,
                             help="Genomic window as chr:start-end[:strand].")
    add_input_arguments(info)
    info.add_argument("--transcript", "-t", metavar="ID", help="Report only this isoform.")
    info.add_argument("--drawn-only", action="store_true",
                      help="Report only the isoforms a figure would draw.")
    info.add_argument("--no-exons", action="store_true",
                      help="Leave out the per-exon listing.")
    info.add_argument("--format", choices=["text", "tsv", "csv"], default="text",
                      help="A readable report, or a table.")
    info.add_argument("--exons-table", action="store_true",
                      help="With --format tsv or csv, one row per exon instead of per isoform.")
    info.add_argument("--output", "-o", metavar="FILE",
                      help="Output file; stdout when omitted.")
    info.add_argument("--quiet", "-q", action="store_true", help="Print no progress.")
    info.set_defaults(func=cmd_info)

    index = subparsers.add_parser(
        "index", help="index a TSV read annotation",
        description="Index a TSV read annotation by gene, so a plot reads one gene instead "
                    "of the whole file.",
        formatter_class=_HelpFormatter,
    )
    index.add_argument("--read-annot", required=True, metavar="TSV",
                       help="Read-to-transcript table to index.")
    index.add_argument("--buckets", type=int, default=64, metavar="N",
                       help="Number of index buckets.")
    index.add_argument("--force", action="store_true", help="Rebuild an up-to-date index.")
    index.add_argument("--drop", action="store_true", help="Delete the index.")
    index.set_defaults(func=cmd_index)

    return parser


def use_file_backend() -> None:
    """Draw with Agg, which only writes files, unless ``MPLBACKEND`` says otherwise."""
    if not os.environ.get("MPLBACKEND"):
        import matplotlib

        matplotlib.use("Agg")


def main(argv: Optional[Sequence[str]] = None) -> int:
    use_file_backend()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"isotracks: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
