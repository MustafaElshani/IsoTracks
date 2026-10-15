"""The gene view: isoform models, read stacks per condition, end sites and coverage.

    from isotracks import IsoTracks

    view = IsoTracks(
        gene_name="SAT1",
        isoforge="results/project",
        bam_files={"Control": ["c1.bam", "c2.bam"], "Treated": ["t1.bam", "t2.bam"]},
    )
    view.plot(output="SAT1")

:func:`plot_gene` does the same in one call.
"""

from __future__ import annotations

import inspect
import os
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam

from . import theme as theme_module
from .theme import MAX_OVERLAID_MODS
from .annotation import (
    ALL_READS,
    SUPPORTED_ONLY,
    GeneReads,
    GtfIndex,
    Locus,
    ReadAnnotation,
    load_bed,
    map_samples_to_conditions,
    select_bed_transcripts,
)
from .canvas import Canvas
from .highlight import (
    HIGHLIGHT_STYLES,
    Highlight,
    draw_highlights,
    normalise_target,
    parse_coordinates,
    split_spec,
)
from .io._intervals import merge_intervals
from .introns import INTRON_GENOMIC, INTRON_MODES, build_intron_map
from .isoforge import find_outputs
from .io.features import BedTranscripts, Feature
from .io.longreads import (
    POLYA_VISUAL_SCALE,
    LongReadBam,
    RegionReadPool,
    read_key,
)
from .io.clusters import ClusterSiteReader
from .io.ends import DEFAULT_BANDWIDTH, ReadEndReader
from .io.moddiff import ModDiffReader, check_diff_tables
from .io.modifications import ModPileupReader, check_pileups, mod_label
from .layout import MM_PER_UNIT, Layout, shorten_label
from .tracks.sites import SiteTrack

BamSpec = Mapping[str, Union[str, Sequence[str]]]

#: Base pairs drawn past the 3' end when ``padding="auto"`` finds no poly(A) tails.
MIN_PADDING = 200
#: Base pairs drawn before the 5' end: four bandwidths of the read-end density.
FIVE_PRIME_PADDING = int(4 * DEFAULT_BANDWIDTH)
#: Extra base pairs past the longest tail, so it never touches the axis edge.
TAIL_CLEARANCE = 60
#: Motif length assumed when sizing the ticks of a track with no motifs.
NOMINAL_MOTIF_LENGTH = 6
#: Tick height as a share of the motif above it.
TICK_LABEL_FRACTION = 0.5
#: Clearance between a site tick and the motif above it, in layout units.
SITE_LABEL_GAP_UNITS = 0.25
#: Modification codes drawn by default.
DEFAULT_MOD_CODES = ("a", "17802")
#: Modified fraction a site must reach to be drawn.
DEFAULT_MOD_FRACTION = 0.3
#: Where the per-position tracks (modifications, differential modification) may draw.
MOD_REGIONS_EXONS = "exons"
MOD_REGIONS_WINDOW = "window"
MOD_REGIONS = (MOD_REGIONS_EXONS, MOD_REGIONS_WINDOW)
#: Which reads the coverage and its sashimi arcs are built from.
COVERAGE_DRAWN = "drawn"
COVERAGE_GENE = "gene"
COVERAGE_READS = (COVERAGE_DRAWN, COVERAGE_GENE)

#: How genes on the opposite strand are drawn: clipped to the window, or whole.
ANTISENSE_WINDOW = "window"
ANTISENSE_WHOLE = "whole"
ANTISENSE = (ANTISENSE_WINDOW, ANTISENSE_WHOLE)
#: Formats written when ``output`` names no extension.
DEFAULT_FORMATS = ("pdf", "png")
#: Columns of :meth:`IsoTracks.isoform_table`.
ISOFORM_TABLE_COLUMNS = [
    "gene", "transcript_id", "chromosome", "strand", "start", "end",
    "exons", "length", "genomic_span", "reads", "median_polya", "drawn",
]
#: Columns of :meth:`IsoTracks.polya_table`.
POLYA_TABLE_COLUMNS = [
    "gene", "transcript_id", "condition", "sample", "read_name", "polya_length",
]
#: A cluster counts as belonging to a transcript when it is this close to one of its ends.
CLUSTER_MATCH_WINDOW = 25
#: Most reads drawn per stack by default.
DEFAULT_MAX_READS = 20_000
#: Most isoforms drawn by default.
DEFAULT_MAX_ISOFORMS = 12


def _one_of(value: str, allowed: Sequence[str], name: str) -> str:
    """Validate a string option at construction, not at draw time."""
    if value not in allowed:
        raise ValueError(
            f"unknown {name} {value!r}; expected one of "
            + ", ".join(repr(a) for a in allowed)
        )
    return value


class BamInputError(ValueError):
    """A BAM file is missing, unreadable, or has no index."""


def normalise_bam_files(bam_files: Optional[BamSpec]) -> Dict[str, List[str]]:
    """``{condition: path}`` or ``{condition: [paths]}`` to ``{condition: [paths]}``."""
    if not bam_files:
        return {}
    out: Dict[str, List[str]] = {}
    for condition, paths in bam_files.items():
        out[condition] = [paths] if isinstance(paths, (str, os.PathLike)) else [str(p) for p in paths]
    return out


def has_index(bam_path: str) -> bool:
    """Whether a coordinate index exists for *bam_path*."""
    stem, _ = os.path.splitext(bam_path)
    return any(os.path.exists(candidate) for candidate in (
        bam_path + ".bai", bam_path + ".csi", stem + ".bai", stem + ".csi",
    ))


def check_bam_files(bam_files: Mapping[str, Sequence[str]]) -> None:
    """Fail early and legibly on missing or unindexed BAMs."""
    missing, unindexed = [], []
    for condition, paths in bam_files.items():
        for path in paths:
            if not os.path.exists(path):
                missing.append(f"  {condition}: {path}")
            elif not has_index(path):
                unindexed.append(f"  {condition}: {path}")

    problems = []
    if missing:
        problems.append("BAM file(s) not found:\n" + "\n".join(missing))
    if unindexed:
        problems.append(
            "BAM file(s) have no index:\n" + "\n".join(unindexed)
            + "\n\nIndex them with:  samtools index <file.bam>"
        )
    if problems:
        raise BamInputError("\n\n".join(problems))


def add_site_tracks(canvas: Canvas, polya: pd.DataFrame, tss: pd.DataFrame, theme,
                    reader_for=None) -> None:
    """poly(A) above TSS, both above the models they annotate."""
    longest = max(_longest_motif(tss), _longest_motif(polya), NOMINAL_MOTIF_LENGTH)
    tick_units = TICK_LABEL_FRACTION * SiteTrack.label_units(
        longest, theme.fonts["annot"], MM_PER_UNIT,
    )

    def source(sites):
        return {"reader": reader_for(sites)} if reader_for else {"fixed_data": sites}

    if not polya.empty:
        height, tick_top = _site_track_metrics(polya, tick_units, theme)
        canvas.add(
            "SiteTrack", label="poly(A)", height=height,
            tick_top=tick_top, label_anchor=tick_top / 2, **source(polya),
        )
    if not tss.empty:
        height, tick_top = _site_track_metrics(tss, tick_units, theme)
        canvas.add(
            "SiteTrack", label="TSS", height=height, tick_color=theme.tss_color,
            tick_top=tick_top, label_anchor=tick_top / 2, **source(tss),
        )


def _longest_motif(sites: pd.DataFrame) -> int:
    motifs = [str(m) for m in sites["motif"].dropna() if m] if not sites.empty else []
    return max((len(m) for m in motifs), default=0)


def _site_track_metrics(sites: pd.DataFrame, tick_units: float, theme
                        ) -> Tuple[float, float]:
    """Track height and the fraction of it the tick occupies."""
    text_units = SiteTrack.label_units(
        _longest_motif(sites), theme.fonts["annot"], MM_PER_UNIT,
    )
    height = tick_units + SITE_LABEL_GAP_UNITS + text_units
    return height, tick_units / height


class IsoTracks:
    """One gene, drawn as a stack of tracks.

    Parameters
    ----------
    gene_name : str, optional
        Gene symbol.  Either this or *region* is required.
    region : tuple, optional
        ``(chromosome, start, end, strand)``, drawn instead of a gene.
    bam_files : mapping
        ``{condition: bam}`` or ``{condition: [bams]}``; replicates are pooled.
    isoforge : str, optional
        An IsoForge run: its output prefix, its directory, or one of its files.
        Fills *annotation_gtf*, *read_annot*, *polya_clusters*, *tss_clusters*,
        *bed_file* (``<prefix>_isoforge_sqanti3.bed``) and *bam_files*
        (``<prefix>_alignments/``, grouped by the sample name up to its first
        underscore); any of those given explicitly takes precedence.
    reference_gtf : str, optional
        Reference annotation, used to place a gene the run's GTF does not contain.
    annotation_gtf : str, optional
        The run's transcript models (``<prefix>_isoforge.gtf``).
    read_annot : str, optional
        Read-to-transcript table, Parquet or TSV
        (``<prefix>_isoforge_read_annot.parquet``).  Without it all reads form one
        stack per condition.
    read_filter : {"supported", "all"}
        ``"supported"`` keeps the whole reads of the transcripts in
        *annotation_gtf* (``cluster_supported``); ``"all"`` keeps every read in
        the table.
    max_frac_a : float, optional
        Also drop reads with no poly(A) signal and a downstream A fraction above this.
    bed_file : str, optional
        BED12 isoform models with coding extent.  Without it models are drawn from
        the GTF exons.
    polya_clusters, tss_clusters : str, optional
        Clustered 3' and 5' end sites (``tes_clusters.bed``, ``tss_clusters.bed``).
    end_density : bool
        Draw the density of read 5' and 3' ends.
    gene_model : bool
        Draw the gene row: the drawn isoforms overlaid in grey.
    modkit_pileup : mapping, optional
        ``{condition: bedMethyl}`` from ``modkit pileup``, bgzipped and tabix-indexed.
    mod_codes : sequence
        Modification codes to draw.  A flat sequence shares one track; nested
        sequences give one track per group.
    min_mod_coverage : int
        Least reads at a site for it to be drawn.
    min_mod_fraction : float
        Least modified fraction at a site for it to be drawn.
    mod_diff : mapping, optional
        ``{label: sites.bed.gz}`` from ``modkit dmr pair``, bgzipped and tabix-indexed.
    max_diff_p_value : float
        Largest p-value of a differential site that is drawn.
    min_diff_effect : float
        Smallest change in modified fraction that is drawn.
    balanced_diff : bool
        Use modkit's replicate-balanced effect size and p-value.
    mod_regions : {"exons", "window"}
        Draw modification sites in the exons of the drawn isoforms, or anywhere in
        the window.
    coverage_reads : {"drawn", "gene"}
        Build coverage from the reads of the drawn isoforms, or from every read of
        the gene.
    antisense : {None, "window", "whole"}
        Also draw the genes transcribed from the opposite strand in the window, each
        with its isoforms, reads and coverage below the gene's own.  ``"whole"``
        widens the window to draw them whole; ``True`` means ``"window"``.
    min_reads : int
        Least reads per isoform, summed over conditions.
    min_reads_per_condition : int, optional
        Least reads per isoform in every condition.
    max_isoforms : int or None
        Most isoforms drawn, best supported first; ``None`` draws all.
    condition_labels : mapping, optional
        Display names, e.g. ``{"Ctrl": "Control"}``.
    condition_colors : mapping, optional
        Colour per condition.
    isoform_colors : sequence, optional
        Colours for the isoforms, in order.
    polya_scale : float
        Base pairs drawn per nucleotide of poly(A) tail.
    theme : Theme, optional
        Visual theme; the active one by default.
    layout : Layout, optional
        Track heights.
    compact : bool
        Use the shorter track heights.  Ignored when *layout* is given.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        gene_name: Optional[str] = None,
        region: Optional[Tuple[str, int, int, str]] = None,
        bam_files: Optional[BamSpec] = None,
        isoforge: Optional[str] = None,
        reference_gtf: Optional[str] = None,
        annotation_gtf: Optional[str] = None,
        read_annot: Optional[str] = None,
        read_filter: str = SUPPORTED_ONLY,
        max_frac_a: Optional[float] = None,
        bed_file: Optional[str] = None,
        polya_clusters: Optional[str] = None,
        tss_clusters: Optional[str] = None,
        end_density: bool = True,
        gene_model: bool = True,
        modkit_pileup: Optional[BamSpec] = None,
        mod_codes: Sequence[str] = DEFAULT_MOD_CODES,
        min_mod_coverage: int = 10,
        min_mod_fraction: float = DEFAULT_MOD_FRACTION,
        mod_diff: Optional[Mapping[str, str]] = None,
        max_diff_p_value: float = 0.05,
        min_diff_effect: float = 0.2,
        balanced_diff: bool = False,
        mod_regions: str = MOD_REGIONS_EXONS,
        coverage_reads: str = COVERAGE_DRAWN,
        antisense: Union[bool, str, None] = None,
        min_reads: int = 5,
        min_reads_per_condition: Optional[int] = None,
        max_isoforms: Optional[int] = DEFAULT_MAX_ISOFORMS,
        condition_labels: Optional[Mapping[str, str]] = None,
        condition_colors: Optional[Mapping[str, str]] = None,
        isoform_colors: Optional[Sequence[str]] = None,
        polya_scale: float = POLYA_VISUAL_SCALE,
        theme: Optional[theme_module.Theme] = None,
        layout: Optional[Layout] = None,
        compact: bool = True,
        verbose: bool = True,
    ):
        if not gene_name and not region:
            raise ValueError("provide either gene_name or region")
        if isoforge:
            found = find_outputs(isoforge)
            bam_files = bam_files or found.get("bam_files")
            annotation_gtf = annotation_gtf or found.get("annotation_gtf")
            read_annot = read_annot or found.get("read_annot")
            bed_file = bed_file or found.get("bed_file")
            polya_clusters = polya_clusters or found.get("polya_clusters")
            tss_clusters = tss_clusters or found.get("tss_clusters")

        self.gene_name = gene_name
        self._explicit_region = tuple(region) if region else None
        self.reference_gtf = reference_gtf
        self.annotation_gtf = annotation_gtf
        self.bed_file = bed_file
        self.polya_clusters = polya_clusters
        self.tss_clusters = tss_clusters
        self.end_density = end_density
        self.gene_model = gene_model
        self.modkit_pileup = normalise_bam_files(modkit_pileup)
        if self.modkit_pileup:
            check_pileups(self.modkit_pileup)
        self.mod_groups = self._normalise_mod_codes(mod_codes, verbose)
        self.min_mod_coverage = min_mod_coverage
        self.min_mod_fraction = min_mod_fraction
        self.mod_diff = dict(mod_diff or {})
        if self.mod_diff:
            check_diff_tables(self.mod_diff)
        self.max_diff_p_value = max_diff_p_value
        self.min_diff_effect = min_diff_effect
        self.balanced_diff = balanced_diff
        self.mod_regions = _one_of(mod_regions, MOD_REGIONS, "mod_regions")
        self.coverage_reads = _one_of(coverage_reads, COVERAGE_READS,
                                      "coverage_reads")
        if antisense is True:
            antisense = ANTISENSE_WINDOW
        self.antisense = _one_of(antisense, ANTISENSE, "antisense") if antisense else None
        self.min_reads = min_reads
        self.min_reads_per_condition = min_reads_per_condition
        self.max_isoforms = max_isoforms
        self.polya_scale = float(polya_scale)
        self.verbose = verbose

        self.bam_files = normalise_bam_files(bam_files)
        if self.bam_files:
            check_bam_files(self.bam_files)

        self.condition_labels = dict(condition_labels or {})
        self.theme = theme or theme_module.active()
        if isoform_colors:
            self.theme = self.theme.derive(isoform_colors=list(isoform_colors))
        self.compact = compact
        self.layout = layout or (Layout.compact() if compact else Layout())

        conditions = list(self.bam_files)
        palette = self.theme.condition_palette(conditions)
        if condition_colors:
            palette.update(condition_colors)
        self.condition_colors = palette

        self._annotation_index = GtfIndex(annotation_gtf) if annotation_gtf else None
        self._reference_index = GtfIndex(reference_gtf) if reference_gtf else None
        self.read_filter = read_filter
        self._read_annotation = (
            ReadAnnotation(read_annot, verbose=verbose, read_filter=read_filter,
                           max_frac_a=max_frac_a)
            if read_annot else None
        )

        self._read_pools: Dict[str, RegionReadPool] = {}
        self._locus: Optional[Locus] = None
        self._gene_reads: Optional[GeneReads] = None
        self._transcripts: Optional[List[str]] = None
        self._isoform_colors: Dict[str, str] = {}
        self._bed_frame: Optional[pd.DataFrame] = None
        self._window: Optional[Tuple] = None
        self._antisense_views: Dict[Tuple[str, int, int], List["IsoTracks"]] = {}
        self._antisense_span: Optional[Tuple] = None
        self.canvas: Optional[Canvas] = None
        self.figure = None

    # ------------------------------------------------------------- logging

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    def read_pool(self, condition: str) -> RegionReadPool:
        """The shared region fetch for one condition's BAMs."""
        pool = self._read_pools.get(condition)
        if pool is None:
            pool = RegionReadPool(self.bam_files[condition], strand_filter=self.strand)
            self._read_pools[condition] = pool
        return pool

    def release_reads(self) -> None:
        """Drop the cached region reads, freeing their memory."""
        for pool in self._read_pools.values():
            pool.clear()

    # ------------------------------------------------------- gene placement

    @property
    def locus(self) -> Locus:
        """Where the gene is."""
        if self._locus is None:
            self._locus = self._resolve_locus()
        return self._locus

    def _resolve_locus(self) -> Locus:
        if self._explicit_region:
            return self._window_locus()

        if not self._annotation_index and not self._reference_index:
            raise ValueError(
                "a gene name needs a GTF: pass reference_gtf and/or annotation_gtf"
            )
        annotated = self._annotated_locus(announce=True)
        if annotated is not None:
            return annotated
        searched = ", ".join(p for p in (self.annotation_gtf, self.reference_gtf) if p)
        raise ValueError(f"gene {self.gene_name!r} not found in {searched}")

    def _annotated_locus(self, announce: bool = False) -> Optional[Locus]:
        """The gene as the annotation places it, or None if it is not there."""
        if not self.gene_name:
            return None
        for index, why in ((self._annotation_index, "annotation GTF"),
                           (self._reference_index, "reference GTF")):
            if index is None:
                continue
            records = index.gene_records(self.gene_name)
            observed = records.transcript_extent()
            if observed is not None:
                if announce:
                    self._say(
                        f"{self.gene_name}: {observed.chromosome}:{observed.start:,}-"
                        f"{observed.end:,} ({observed.strand}), from the {why}"
                    )
                return observed
            if records.gene is not None:
                return records.gene
        return None

    def _window_locus(self) -> Locus:
        """A window given as ``region=``, oriented by what is actually there."""
        chrom, start, end, *rest = self._explicit_region
        given = rest[0] if rest and rest[0] in ("+", "-") else None
        label = self.gene_name or f"{chrom}:{int(start)}-{int(end)}"

        annotated = self._annotated_locus() if self.gene_name else None
        if annotated is not None:
            if annotated.chromosome != chrom:
                raise ValueError(
                    f"{self.gene_name} is on {annotated.chromosome}, but the window "
                    f"is on {chrom}"
                )
            if given is not None and given != annotated.strand:
                self._say(
                    f"  {self.gene_name} is on the {annotated.strand} strand; the "
                    f"window said {given}.  Drawing it {annotated.strand}, 5' to 3' "
                    "left to right, and keeping its reads"
                )
            strand = annotated.strand
        elif given is not None:
            strand = given
        else:
            strand = self._infer_strand(chrom, int(start), int(end))
        return Locus(label, chrom, int(start), int(end), strand, source="region")

    #: Reads sampled to infer a window's strand.
    STRAND_SAMPLE = 5_000

    def _infer_strand(self, chrom: str, start: int, end: int) -> str:
        """The strand most reads in the window align to."""
        counts = {"+": 0, "-": 0}
        seen = 0
        for paths in self.bam_files.values():
            for path in paths:
                try:
                    with pysam.AlignmentFile(path, "rb") as handle:
                        for read in handle.fetch(chrom, max(start - 1, 0), end):
                            if read.is_unmapped or read.is_secondary \
                                    or read.is_supplementary:
                                continue
                            counts["-" if read.is_reverse else "+"] += 1
                            seen += 1
                            if seen >= self.STRAND_SAMPLE:
                                break
                except (OSError, ValueError):
                    continue
                if seen >= self.STRAND_SAMPLE:
                    break
            if seen >= self.STRAND_SAMPLE:
                break

        if not seen:
            self._say(
                f"  no strand given for {chrom}:{start:,}-{end:,} and no reads to "
                "infer one from; drawing it as + (5' left).  Append :- for a "
                "minus-strand locus"
            )
            return "+"
        strand = "-" if counts["-"] > counts["+"] else "+"
        self._say(
            f"  no strand given for {chrom}:{start:,}-{end:,}; {counts['+']:,} "
            f"reads on +, {counts['-']:,} on -, so drawing it {strand}, 5' to 3' "
            "left to right"
        )
        return strand

    @property
    def region(self) -> Tuple[str, int, int, str]:
        """The gene's own extent, before any display padding."""
        return self.locus.region

    @property
    def chromosome(self) -> str:
        return self.locus.chromosome

    @property
    def strand(self) -> str:
        return self.locus.strand

    # ------------------------------------------------------ read annotation

    @property
    def gene_reads(self) -> GeneReads:
        """The read-to-transcript rows for this gene."""
        if self._gene_reads is None:
            if self._read_annotation is None:
                self._gene_reads = GeneReads(self.gene_name or "", pd.DataFrame(), {})
            else:
                locus = self.locus
                reads = self._read_annotation.for_gene(
                    self.gene_name, locus.chromosome, locus.strand,
                )
                reads = self._restrict_to_models(reads)
                self._gene_reads = reads
                self._say(
                    f"read annotation: {len(reads.read_names):,} reads across "
                    f"{len(reads.reads_by_transcript)} transcripts"
                )
        return self._gene_reads

    def model_transcripts(self) -> Set[str]:
        """Transcript ids the annotation GTF holds for this gene."""
        if not self._annotation_index or not self.gene_name:
            return set()
        frame = self._annotation_index.gene_records(self.gene_name).transcripts
        if frame.empty:
            return set()
        return {str(tid) for tid in frame["transcript_id"] if tid}

    def drawn_exons(self, transcripts: Optional[Sequence[str]] = None
                    ) -> List[Tuple[int, int]]:
        """Union of the exons of the isoforms this figure draws."""
        if not self._annotation_index or not self.gene_name:
            return []
        wanted = set(transcripts if transcripts is not None else self.transcripts())
        if not wanted:
            return []
        exons = self._annotation_index.gene_records(self.gene_name).exons
        return merge_intervals(
            span for tid, spans in exons.items() if str(tid) in wanted
            for span in spans
        )

    def _restrict_to_models(self, reads: GeneReads) -> GeneReads:
        """Drop reads on transcripts the annotation GTF does not carry."""
        if self.read_filter == ALL_READS:
            return reads
        models = self.model_transcripts()
        if not models:
            if self._annotation_index:
                self._say(
                    "  no transcripts for this gene in the annotation GTF; "
                    "every read in the table is kept"
                )
            return reads
        kept = reads.restrict_to(models)
        dropped = len(reads.reads_by_transcript) - len(kept.reads_by_transcript)
        if dropped:
            self._say(
                f"  dropped {dropped} transcript(s) absent from the annotation GTF "
                "(read_filter='all' keeps them)"
            )
        return kept

    def transcripts(self, refresh: bool = False) -> List[str]:
        """Isoforms that clear the read thresholds, in a stable order."""
        if self._transcripts is not None and not refresh:
            return self._transcripts

        counts = self.gene_reads.counts()
        if not counts:
            kept: List[str] = []
        elif self.min_reads_per_condition is not None and self.bam_files:
            kept = self._filter_per_condition(counts)
        else:
            kept = sorted(tid for tid, count in counts.items() if count >= self.min_reads)

        self._transcripts = self._cap_isoforms(kept, counts)
        self._isoform_colors = self.theme.isoform_palette(self._transcripts)
        return self._transcripts

    def _cap_isoforms(self, kept: List[str], counts: Mapping[str, int]) -> List[str]:
        """Keep the best-supported isoforms when there are too many to read."""
        limit = self.max_isoforms
        if limit is None or len(kept) <= limit:
            return kept
        ranked = sorted(kept, key=lambda tid: (-counts.get(tid, 0), tid))
        chosen = sorted(ranked[:limit])
        self._say(
            f"  {len(kept)} isoforms passed; drawing the {limit} best supported "
            "(max_isoforms=None draws all)"
        )
        return chosen

    def _filter_per_condition(self, counts: Dict[str, int]) -> List[str]:
        """Keep isoforms seen at least *min_reads_per_condition* times everywhere."""
        threshold = self.min_reads_per_condition
        sample_map = map_samples_to_conditions(self.bam_files, self.gene_reads.samples)
        per_condition = self.gene_reads.counts_by_condition(sample_map)

        if per_condition.empty:
            self._say(
                "  no usable sample column in the read annotation; "
                f"falling back to a total of >={self.min_reads} reads"
            )
            return sorted(tid for tid, count in counts.items() if count >= self.min_reads)

        table = per_condition.pivot_table(
            index="transcript_id", columns="condition", values="count",
            aggfunc="sum", fill_value=0,
        )
        conditions = list(self.bam_files)
        for condition in conditions:
            if condition not in table.columns:
                table[condition] = 0

        kept: List[str] = []
        for tid in sorted(counts):
            row = table.loc[tid] if tid in table.index else None
            per = {c: int(row[c]) if row is not None else 0 for c in conditions}
            passes = all(value >= threshold for value in per.values())
            if passes:
                kept.append(tid)
            if self.verbose:
                mark = "keep" if passes else "drop"
                detail = ", ".join(
                    f"{self.condition_labels.get(c, c)}={per[c]}" for c in conditions
                )
                print(f"  [{mark}] {tid}: {detail} (total={sum(per.values())})")

        self._say(
            f"  {len(kept)}/{len(counts)} isoforms have >={threshold} reads in every condition"
        )
        return kept

    def isoform_color(self, transcript_id: str) -> str:
        if not self._isoform_colors:
            self.transcripts()
        return self._isoform_colors.get(transcript_id, self.theme.isoform_color(0))

    # ------------------------------------------------------------- isoforms

    def bed_frame(self) -> pd.DataFrame:
        """BED12 rows for the selected isoforms, keyed by transcript id."""
        if self._bed_frame is not None:
            return self._bed_frame
        if not self.bed_file:
            self._bed_frame = pd.DataFrame()
            return self._bed_frame

        frame = load_bed(self.bed_file)
        name_map = (
            self._annotation_index.transcript_name_map(self.gene_name)
            if self._annotation_index and self.gene_name else {}
        )
        self._bed_frame = select_bed_transcripts(frame, self.transcripts(), name_map)
        return self._bed_frame

    def exons(self, transcript_id: Optional[str] = None,
              all_isoforms: bool = False) -> pd.DataFrame:
        """Exons of this gene, numbered 5' to 3' in transcription order."""
        index = self._annotation_index or self._reference_index
        if index is None or not self.gene_name:
            return pd.DataFrame(columns=["transcript_id", "exon_number", "chromosome",
                                         "start", "end", "length", "strand", "position"])
        records = index.gene_records(self.gene_name)
        by_transcript = records.exons_in_transcription_order(transcript_id)
        keep = (set(by_transcript) if all_isoforms
                else set(self.transcripts() or by_transcript))

        rows = []
        for tid, spans in by_transcript.items():
            if tid not in keep and transcript_id is None:
                continue
            for number, (start, end) in enumerate(spans, 1):
                rows.append({
                    "transcript_id": tid,
                    "exon_number": number,
                    "chromosome": self.chromosome,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,  # GTF spans are inclusive
                    "strand": self.strand,
                    "position": "5_prime" if number == 1
                    else ("3_prime" if number == len(spans) else "internal"),
                })
        return pd.DataFrame(rows)

    def isoform_table(self, transcript_id: Optional[str] = None,
                      all_isoforms: bool = False) -> pd.DataFrame:
        """One row per isoform: how many exons, how long, how well supported."""
        exons = self.exons(transcript_id, all_isoforms=all_isoforms)
        if exons.empty:
            return pd.DataFrame(columns=ISOFORM_TABLE_COLUMNS)

        counts = self.gene_reads.counts()
        tails = self.gene_reads.polya_lengths()
        reads_by_transcript = self.gene_reads.reads_by_transcript
        drawn = set(self.transcripts())

        rows = []
        for tid, group in exons.groupby("transcript_id", sort=False):
            tail_lengths = [
                tails[name] for name in reads_by_transcript.get(tid, ())
                if name in tails
            ]
            rows.append({
                "gene": self.gene_name or "",
                "transcript_id": tid,
                "chromosome": self.chromosome,
                "strand": self.strand,
                "start": int(group["start"].min()),
                "end": int(group["end"].max()),
                "exons": int(len(group)),
                "length": int(group["length"].sum()),  # mature
                "genomic_span": int(group["end"].max() - group["start"].min() + 1),
                "reads": int(counts.get(tid, 0)),
                "median_polya": (float(np.median(tail_lengths))
                                 if tail_lengths else float("nan")),
                "drawn": tid in drawn,
            })
        table = pd.DataFrame(rows, columns=ISOFORM_TABLE_COLUMNS)
        return table.sort_values("reads", ascending=False).reset_index(drop=True)

    def describe(self, transcript_id: Optional[str] = None,
                 exons: bool = True, all_isoforms: bool = True,
                 width: int = 78) -> str:
        """The gene, its isoforms and their exons as text meant to be read."""
        isoforms = self.isoform_table(transcript_id, all_isoforms=all_isoforms)
        lines = [
            "=" * width,
            f"{self.gene_name or 'region'}   {self.chromosome}:"
            f"{self.locus.start:,}-{self.locus.end:,}   {self.strand} strand",
            "=" * width,
        ]
        if isoforms.empty:
            lines.append("no isoforms found for this gene in the annotation")
            return "\n".join(lines)

        total_reads = int(isoforms["reads"].sum())
        lines += [
            f"{len(isoforms)} isoform(s), {total_reads:,} assigned read(s)",
            "Coordinates are 1-based and inclusive; length is the mature",
            "transcript (exons summed), span is first base to last.",
            "",
            f"{'isoform':<24}{'exons':>6}{'length':>9}{'span':>10}"
            f"{'reads':>9}{'polyA':>8}  ",
        ]
        for row in isoforms.itertuples(index=False):
            tail = "-" if row.median_polya != row.median_polya else f"{row.median_polya:.0f}"
            mark = "" if row.drawn else "  (not drawn)"
            lines.append(
                f"{shorten_label(row.transcript_id, 23):<24}{row.exons:>6}"
                f"{row.length:>9,}{row.genomic_span:>10,}{row.reads:>9,}"
                f"{tail:>8}{mark}"
            )

        if exons:
            lines += ["", "Exons, numbered 5' to 3' in transcription order:"]
            table = self.exons(transcript_id, all_isoforms=all_isoforms)
            for tid, group in table.groupby("transcript_id", sort=False):
                lines += ["", f"  {tid}"]
                for exon in group.itertuples(index=False):
                    lines.append(
                        f"    exon {exon.exon_number:>3}  "
                        f"{exon.start:>12,} - {exon.end:>12,}  "
                        f"{exon.length:>7,} bp   {exon.position}"
                    )
        return "\n".join(lines)

    def print_info(self, transcript_id: Optional[str] = None,
                   exons: bool = True, all_isoforms: bool = True) -> None:
        """Print :meth:`describe`."""
        print(self.describe(transcript_id, exons=exons, all_isoforms=all_isoforms))

    def zoom(self, target: str = "last_exon", transcript_id: Optional[str] = None,
             padding: int = 200, tail_room: bool = True) -> Tuple[str, int, int, str]:
        """A region to pass to :meth:`plot` as ``region=``."""
        chrom, gene_start, gene_end, strand = self.region

        if target == "whole_gene":
            return self._with_tail_room(
                (chrom, gene_start - padding, gene_end + padding, strand), tail_room)
        if target in ("5_prime", "3_prime"):
            window = 2000
            at_low_end = (target == "5_prime") == (strand == "+")
            if at_low_end:
                region = (chrom, gene_start - padding,
                          min(gene_end, gene_start + window), strand)
            else:
                region = (chrom, max(gene_start, gene_end - window),
                          gene_end + padding, strand)
            return self._with_tail_room(region, tail_room and target == "3_prime")

        chosen, rows = self._reference_exons(transcript_id)

        if target == "first_exon":
            exon = rows.iloc[0]
        elif target == "last_exon":
            exon = rows.iloc[-1]
        elif target.startswith("exon_"):
            try:
                number = int(target.split("_", 1)[1])
            except ValueError:
                raise ValueError(f"bad exon target {target!r}; use e.g. 'exon_3'") from None
            match = rows[rows["exon_number"] == number]
            if match.empty:
                raise ValueError(
                    f"{chosen} has {len(rows)} exons, so exon {number} does not exist"
                )
            exon = match.iloc[0]
        else:
            raise ValueError(
                f"unknown zoom target {target!r}; use 'first_exon', 'last_exon', "
                "'exon_<N>', '5_prime', '3_prime' or 'whole_gene'"
            )

        region = (chrom, int(exon["start"]) - padding, int(exon["end"]) + padding, strand)
        at_three_prime = target == "last_exon" or (
            target.startswith("exon_") and int(exon["exon_number"]) == len(rows)
        )
        return self._with_tail_room(region, tail_room and at_three_prime)

    def _intron_map(self, window: Sequence, mode: str, share: float):
        """The intron-shortening map for *window*, or None to draw true scale."""
        if mode == INTRON_GENOMIC:
            return None
        exons = self.drawn_exons(self.transcripts() or sorted(self.model_transcripts()))
        if self.antisense:
            for view in self.antisense_views(self._antisense_span or window):
                exons = merge_intervals(list(exons) + view.drawn_exons())
        intron_map = build_intron_map(exons, window, mode, share)
        if intron_map is None:
            self._say(f"  introns={mode!r}: nothing to shorten in this window, "
                      "so it is drawn at true scale")
            return None
        genomic = sum(end - start + 1 for start, end in intron_map.breaks)
        drawn = sum(float(intron_map.forward(end + 0.5) - intron_map.forward(start - 0.5))
                    for start, end in intron_map.breaks)
        self._say(f"  introns {mode}: {len(intron_map.breaks)} shortened, "
                  f"{genomic:,} bp drawn in the width of {drawn:,.0f}")
        return intron_map

    def detail_region(self, target, transcript_id: Optional[str] = None,
                      padding: int = 200) -> Tuple[Tuple[str, int, int, str], Highlight]:
        """The window a detail panel shows, and the marker that names it."""
        if isinstance(target, str) and normalise_target(split_spec(target)[0]) in (
                "5_prime", "3_prime"):
            name = normalise_target(split_spec(target)[0])
            window = self.zoom(name, padding=padding)
            label = {"5_prime": "5′ end", "3_prime": "3′ end"}[name]
            return window, Highlight([(window[1], window[2])], label=label,
                                     style="outline")

        if isinstance(target, str):
            name, named_transcript = split_spec(target)
            region = self.highlight_region(name, transcript_id or named_transcript)
        else:
            region = self.highlight_region(target, transcript_id)

        chrom, gene_start, gene_end, strand = self.region
        start, end = region.start - int(padding), region.end + int(padding)
        reaches_three_prime = (end >= gene_end if strand != "-"
                               else start <= gene_start)
        window = self._with_tail_room((chrom, start, end, strand), reaches_three_prime)
        marker = Highlight([(window[1], window[2])], label=region.label,
                           label_color=region.label_color,
                           transcript_id=region.transcript_id, style="outline")
        return window, marker

    def _oriented(self, region: Sequence) -> Tuple[str, int, int, str]:
        """A window to draw, on this view's strand whatever it was given as."""
        chrom, start, end, *rest = tuple(region)
        given = rest[0] if rest else None
        if given is not None and given != self.strand:
            self._say(
                f"  the window said {given}, but this view is on the {self.strand} "
                f"strand; drawing it {self.strand} so it reads 5' to 3', left to right"
            )
        return (chrom, int(start), int(end), self.strand)

    # ------------------------------------------------------------ highlights

    def highlight_region(self, target: Union[str, Tuple[int, int]],
                         transcript_id: Optional[str] = None) -> Highlight:
        """Resolve a region to mark into concrete genomic spans."""
        if isinstance(target, (tuple, list)) and len(target) == 2 \
                and all(isinstance(v, (int, np.integer)) for v in target):
            start, end = sorted(int(v) for v in target)
            return Highlight([(start, end)])
        if not isinstance(target, str):
            raise TypeError(f"cannot highlight {target!r}; give a name or (start, end)")

        coordinates = parse_coordinates(target)
        if coordinates is not None:
            chrom, start, end = coordinates
            if chrom and chrom != self.chromosome:
                raise ValueError(
                    f"highlight {target!r} is on {chrom}, but this figure is on "
                    f"{self.chromosome}"
                )
            return Highlight([(start, end)])

        name = normalise_target(target)
        if name in ("5_utr", "3_utr", "cds"):
            return self._coding_highlight(name, transcript_id)

        chosen, rows = self._reference_exons(transcript_id)
        spans = [(int(r.start), int(r.end)) for r in rows.itertuples(index=False)]
        count = len(spans)

        if name in ("first_exon", "last_exon") or name.startswith("exon_"):
            number = (1 if name == "first_exon" else count if name == "last_exon"
                      else self._region_number(name, "exon"))
            if not 1 <= number <= count:
                raise ValueError(f"{chosen} has {count} exons, so exon {number} "
                                 "does not exist")
            region, label = [spans[number - 1]], f"exon {number}"
        elif name.startswith(("intron_", "junction_")):
            kind = name.split("_", 1)[0]
            number = self._region_number(name, kind)
            if not 1 <= number < count:
                raise ValueError(
                    f"{chosen} has {count} exons and so {count - 1} introns; "
                    f"there is no {kind} {number}"
                )
            first, second = sorted((spans[number - 1], spans[number]))
            region = [(first[1] + 1, second[0] - 1)]
            label = (f"intron {number}" if kind == "intron"
                     else f"junction {number}–{number + 1}")
        else:
            raise ValueError(
                f"unknown highlight {target!r}; use 'exon_<N>', 'first_exon', "
                "'last_exon', 'intron_<N>', 'junction_<N>', '5_utr', '3_utr', "
                "'cds', or a range such as '31950000-31951000'"
            )
        return self._named_highlight(region, label, chosen)

    @staticmethod
    def _region_number(name: str, kind: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(f"bad {kind} target {name!r}; use e.g. '{kind}_2'") from None

    def _named_highlight(self, spans, label: str, transcript_id: str) -> Highlight:
        drawn = transcript_id in set(self.transcripts())
        return Highlight(
            spans=[(int(a), int(b)) for a, b in spans], label=label,
            label_color=self.isoform_color(transcript_id) if drawn else None,
            transcript_id=transcript_id,
        )

    def _coding_highlight(self, name: str, transcript_id: Optional[str]) -> Highlight:
        """A UTR or the CDS, from the BED's thickStart/thickEnd."""
        label = {"5_utr": "5′ UTR", "3_utr": "3′ UTR", "cds": "CDS"}[name]
        if not self.bed_file:
            raise ValueError(f"a {label} highlight needs a BED with coding "
                             "coordinates; pass bed_file (SQANTI3's carries them)")

        rows = self._bed_rows_for_gene()
        if transcript_id:
            candidates = [transcript_id]
        else:
            order = rows.sort_values("blockCount", ascending=False)["name"].tolist()
            preferred = self._reference_exons()[0]
            candidates = [preferred] + [t for t in order if t != preferred]

        for tid in candidates:
            match = rows[rows["name"] == tid]
            if match.empty:
                continue
            spans = _coding_spans(match.iloc[0], name)
            if spans:
                return self._named_highlight(spans, label, tid)
            if transcript_id:
                raise ValueError(f"{tid} has no {label}: its BED record has no "
                                 "coding region there")
        if transcript_id:
            raise ValueError(f"{transcript_id} is not in {os.path.basename(self.bed_file)}")
        raise ValueError(f"no isoform of {self.gene_name} has a {label} in "
                         f"{os.path.basename(self.bed_file)}")

    def _bed_rows_for_gene(self) -> pd.DataFrame:
        """Every BED12 record for this gene's annotated isoforms, by id."""
        wanted = self.model_transcripts() or set(self.transcripts())
        name_map = (self._annotation_index.transcript_name_map(self.gene_name)
                    if self._annotation_index and self.gene_name else {})
        return select_bed_transcripts(load_bed(self.bed_file), sorted(wanted), name_map)

    def _resolve_highlights(self, highlight) -> List[Highlight]:
        """Turn whatever ``plot(highlight=...)`` was given into Highlights."""
        if highlight is None:
            return []
        items = ([highlight] if isinstance(highlight, (str, Highlight))
                 or (isinstance(highlight, tuple) and len(highlight) == 2
                     and all(isinstance(v, (int, np.integer)) for v in highlight))
                 else list(highlight))
        resolved = []
        for item in items:
            if isinstance(item, Highlight):
                resolved.append(item)
                continue
            if isinstance(item, str):
                target, transcript = split_spec(item)
                region = self.highlight_region(target, transcript)
            else:
                region = self.highlight_region(item)
            where = ", ".join(f"{a:,}-{b:,}" for a, b in region.spans)
            source = f" on {region.transcript_id}" if region.transcript_id else ""
            self._say(f"  highlight {region.label or 'region'}{source}: "
                      f"{self.chromosome}:{where}")
            resolved.append(region)
        return resolved

    def _reference_exons(self, transcript_id: Optional[str] = None
                         ) -> Tuple[str, pd.DataFrame]:
        """The isoform named regions are measured on, and its exons in order."""
        exons = self.exons(transcript_id)
        if exons.empty:
            raise ValueError(f"no exons available for {self.gene_name!r}; need a GTF")
        chosen = transcript_id or (
            exons.groupby("transcript_id")["exon_number"].max().idxmax()
        )
        rows = exons[exons["transcript_id"] == chosen].sort_values("exon_number")
        if rows.empty:
            raise ValueError(f"transcript {chosen!r} has no exons here")
        return str(chosen), rows

    def _with_tail_room(self, region: Tuple[str, int, int, str],
                        wanted: bool) -> Tuple[str, int, int, str]:
        """Widen a window's 3' side to fit the longest poly(A) tail."""
        if not wanted or not self.bam_files:
            return region
        longest = self.max_polya_length()
        if longest <= 0:
            return region
        room = int(longest * self.polya_scale) + TAIL_CLEARANCE
        chrom, start, end, strand = region
        if strand == "+":
            return (chrom, start, end + room, strand)
        return (chrom, max(1, start - room), end, strand)

    # -------------------------------------------------------- display region

    def _transcript_bounds(self) -> Optional[Tuple[int, int]]:
        """Extent of the isoforms actually being drawn."""
        transcripts = set(self.transcripts())
        if self._annotation_index and self.gene_name:
            frame = self._annotation_index.gene_records(self.gene_name).transcripts
            if not frame.empty and transcripts:
                selected = frame[frame["transcript_id"].isin(transcripts)]
                if not selected.empty:
                    return int(selected["start"].min()), int(selected["end"].max())

        bed = self.bed_frame()
        if not bed.empty:
            return int(bed["chromStart"].min()) + 1, int(bed["chromEnd"].max())
        return None

    def max_polya_length(self) -> int:
        """Longest poly(A) tail among the reads that will be drawn."""
        if not self.bam_files:
            return 0
        allowed = self.gene_reads.read_names or None
        fallback = self.gene_reads.polya_lengths()
        longest = 0
        for condition in self.bam_files:
            reader = LongReadBam(
                pool=self.read_pool(condition), allowed_read_names=allowed,
                polya_lengths=fallback, strand_filter=self.strand,
                polya_scale=self.polya_scale,
            )
            longest = max(longest, reader.max_polya_length(self.region))
        return longest

    def display_region(self, padding: Union[int, str] = "auto") -> Tuple[str, int, int, str]:
        """The window to draw, with room at the 3' end for poly(A) tails."""
        if padding == "auto":
            longest = self.max_polya_length()
            padding = (
                max(MIN_PADDING, int(longest * self.polya_scale) + TAIL_CLEARANCE)
                if longest else MIN_PADDING
            )
            self._say(f"3' padding: {padding:,} bp (longest poly(A) tail {longest} nt)")
        padding = int(padding)

        bounds = self._transcript_bounds()
        chrom, gene_start, gene_end, strand = self.region
        start, end = bounds if bounds else (gene_start, gene_end)

        if strand == "+":
            window = (chrom, max(1, start - FIVE_PRIME_PADDING), end + padding, strand)
        else:
            window = (chrom, max(1, start - padding), end + FIVE_PRIME_PADDING, strand)
        if self.antisense == ANTISENSE_WHOLE:
            self._antisense_span = window
            for view in self.antisense_views(window):
                _, low, high, _ = view.display_region(padding)
                window = (chrom, min(window[1], low), max(window[2], high), strand)
        return window

    # ------------------------------------------------------ opposite strand

    def antisense_views(self, span: Sequence) -> List["IsoTracks"]:
        """A view of each gene on the opposite strand with isoforms drawn in *span*."""
        key = (str(span[0]), int(span[1]), int(span[2]))
        if key in self._antisense_views:
            return self._antisense_views[key]
        index = self._annotation_index or self._reference_index
        opposite = "-" if self.strand == "+" else "+"
        names = [name for name in index.genes_overlapping(*key, strand=opposite)
                 if name != self.gene_name] if index else []
        views = []
        for name in names:
            view = self._opposite_view(name)
            if self.antisense == ANTISENSE_WINDOW:
                view.keep_transcripts_in(key)
            if view.transcripts():
                views.append(view)
        drawn = list(self.transcripts()) + [t for v in views for t in v.transcripts()]
        palette = self.theme.isoform_palette(drawn)
        for view in views:
            view._isoform_colors = {t: palette[t] for t in view.transcripts()}
        self._say(f"  opposite strand: {', '.join(v.gene_name for v in views) or 'nothing drawn'}"
                  + (f" (of {', '.join(names)})" if len(names) > len(views) else ""))
        self._antisense_views[key] = views
        return views

    def _opposite_view(self, gene_name: str) -> "IsoTracks":
        """A view of *gene_name* with this view's inputs and settings."""
        view = IsoTracks(
            gene_name=gene_name, bam_files=self.bam_files,
            reference_gtf=self.reference_gtf, annotation_gtf=self.annotation_gtf,
            bed_file=self.bed_file, read_filter=self.read_filter,
            coverage_reads=self.coverage_reads, min_reads=self.min_reads,
            min_reads_per_condition=self.min_reads_per_condition,
            max_isoforms=self.max_isoforms, condition_labels=self.condition_labels,
            condition_colors=self.condition_colors, polya_scale=self.polya_scale,
            theme=self.theme, layout=self.layout, verbose=self.verbose,
        )
        view._read_annotation = self._read_annotation
        return view

    def keep_transcripts_in(self, span: Sequence) -> None:
        """Draw only the isoforms that overlap *span*."""
        frame = (self._annotation_index.gene_records(self.gene_name).transcripts
                 if self._annotation_index else pd.DataFrame())
        if frame.empty:
            return
        low, high = int(span[1]), int(span[2])
        inside = set(frame.loc[(frame["start"] <= high) & (frame["end"] >= low), "transcript_id"])
        self._transcripts = [t for t in self.transcripts() if t in inside]

    def _add_strand_section(self, canvas: Canvas, conditions: Sequence[str],
                            labels: Mapping[str, str], sort_style: str,
                            max_reads: Optional[int]) -> None:
        """This gene below a rule: its gene row, isoforms, reads and coverage."""
        transcripts = self.transcripts()
        features = self.gene_model_features(transcripts)
        if features:
            canvas.add(
                "GeneModelTrack", label=f"{self.gene_name} ({self.strand})",
                reader=BedTranscripts(features=features), height=self.layout.gene_model,
                label_weight="bold", rule_above=True,
            )
        self._add_transcript_tracks(canvas, transcripts)
        self._add_read_tracks(
            canvas, transcripts, conditions, labels, self.gene_reads.reads_by_transcript,
            self.gene_reads.polya_lengths(), sort_style, max_reads,
        )
        self._add_coverage_tracks(canvas, conditions, labels,
                                  self._coverage_read_names(transcripts))

    def polya_sites(self) -> pd.DataFrame:
        """Poly(A) clusters matching a 3' end of one of the drawn isoforms."""
        return self._clusters_near_ends(self.polya_clusters, "polya_site", require_motif=True)

    def tss_sites(self) -> pd.DataFrame:
        """TSS clusters matching a 5' end of one of the drawn isoforms."""
        return self._clusters_near_ends(self.tss_clusters, "tss_site", require_motif=False)

    def _clusters_near_ends(self, path: Optional[str], attribute: str,
                            require_motif: bool) -> pd.DataFrame:
        """Clusters lying within :data:`CLUSTER_MATCH_WINDOW` of a drawn end."""
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        if not self._annotation_index or not self.gene_name:
            return pd.DataFrame()

        frame = self._annotation_index.gene_records(self.gene_name).transcripts
        transcripts = set(self.transcripts())
        if frame.empty or not transcripts or attribute not in frame.columns:
            return pd.DataFrame()
        selected = frame[frame["transcript_id"].isin(transcripts)]
        if selected.empty:
            return pd.DataFrame()

        ends = {int(v) for v in selected[attribute].dropna()}
        chrom = str(selected["chromosome"].iloc[0])
        strand = str(selected["strand"].iloc[0])
        window = (
            chrom,
            max(1, int(selected["start"].min()) - CLUSTER_MATCH_WINDOW),
            int(selected["end"].max()) + CLUSTER_MATCH_WINDOW,
            strand,
        )

        sites = ClusterSiteReader(path, require_motif=require_motif).read_region(window)
        if sites.empty or not ends:
            return sites.iloc[0:0]
        sites = sites[sites["strand"] == strand]
        near = sites["site"].apply(
            lambda site: any(abs(int(site) - end) <= CLUSTER_MATCH_WINDOW for end in ends)
        )
        return sites[near].reset_index(drop=True)

    # ------------------------------------------------------------- building

    def build(self, sort_style: str = "waterfall",
              max_reads: Optional[int] = DEFAULT_MAX_READS,
              window: Optional[Sequence] = None) -> Canvas:
        """Assemble the canvas without drawing it."""
        self._window = tuple(window) if window else self.display_region()
        transcripts = self.transcripts()
        conditions = list(self.bam_files)
        labels = {c: self.condition_labels.get(c, c) for c in conditions}
        reads_by_transcript = self.gene_reads.reads_by_transcript
        coverage_reads = self._coverage_read_names(transcripts)
        polya_fallback = self.gene_reads.polya_lengths()

        opposite = (self.antisense_views(self._antisense_span or self._window)
                    if self.antisense else [])
        gutter_labels = ([shorten_label(t) for t in transcripts] + list(labels.values())
                         + list(self.mod_diff) + ["read ends", "poly(A)"]
                         + [shorten_label(t) for v in opposite for t in v.transcripts()])
        bold = ([self.gene_name] if self.gene_name else []) + [
            f"{v.gene_name} ({v.strand})" for v in opposite]
        gutter_in = self.layout.gutter_width_for(gutter_labels, bold=bold)
        canvas = Canvas(gutter_in=gutter_in)

        self._add_site_tracks(canvas)
        self._add_end_density_track(canvas)
        self._add_gene_model_track(canvas, transcripts)
        self._add_transcript_tracks(canvas, transcripts)
        self._add_read_tracks(
            canvas, transcripts, conditions, labels,
            reads_by_transcript, polya_fallback, sort_style, max_reads,
        )
        self._add_modification_tracks(canvas, conditions, labels)
        self._add_mod_diff_tracks(canvas)
        self._add_coverage_tracks(canvas, conditions, labels, coverage_reads)
        for view in opposite:
            view._add_strand_section(canvas, conditions, labels, sort_style, max_reads)
        canvas.add("ScaleBarTrack", height=self.layout.scale_bar)
        canvas.add("CoordinateAxisTrack", height=self.layout.axis,
                   gene=self.gene_name or "")

        self.canvas = canvas
        return canvas

    def _in_window(self, sites: pd.DataFrame) -> pd.DataFrame:
        """Sites that fall inside the window being drawn."""
        if sites.empty or not self._window:
            return sites
        _chrom, start, end, _strand = self._window
        low, high = sorted((int(start), int(end)))
        return sites[(sites["site"] >= low) & (sites["site"] <= high)]

    def _add_site_tracks(self, canvas: Canvas) -> None:
        add_site_tracks(canvas, self._in_window(self.polya_sites()),
                        self._in_window(self.tss_sites()), self.theme)

    def _add_end_density_track(self, canvas: Canvas) -> None:
        """Where the reads start and stop, under the clusters called from them."""
        if not self.end_density or not self.bam_files:
            return
        reader = ReadEndReader(
            pools=[self.read_pool(condition) for condition in self.bam_files],
            allowed_read_names=self.gene_reads.read_names or None,
        )
        canvas.add(
            "EndDensityTrack", label="read ends", reader=reader,
            height=self.layout.end_density,
        )

    def gene_model_features(self, transcripts: Optional[Sequence[str]] = None):
        """Features for the gene model row: the isoforms the figure draws."""
        drawn = list(transcripts) if transcripts is not None else self.transcripts()
        wanted = set(drawn) or self.model_transcripts()
        if self.bed_file:
            frame = load_bed(self.bed_file)
            name_map = (
                self._annotation_index.transcript_name_map(self.gene_name)
                if self._annotation_index and self.gene_name else {}
            )
            rows = select_bed_transcripts(frame, sorted(wanted), name_map)
            if not rows.empty:
                return _bed_rows_to_features(rows)

        return self._gtf_features(wanted)

    def _gtf_features(self, wanted: Set[str]):
        """Transcript features from the annotation GTF's exons: all of the gene's, or *wanted*."""
        index = self._annotation_index or self._reference_index
        if index is None or not self.gene_name:
            return []
        records = index.gene_records(self.gene_name)
        locus = self.locus
        features = []
        for tid, spans in records.exons.items():
            if wanted and str(tid) not in wanted:
                continue
            if not spans:
                continue
            ordered = sorted(spans)
            parent = Feature(
                ordered[0][0], ordered[-1][1], chr_name=locus.chromosome,
                strand=locus.strand, feature_id=str(tid), type_name="mRNA",
                children=[
                    Feature(start, end, chr_name=locus.chromosome,
                            strand=locus.strand, type_name="exon")
                    for start, end in ordered
                ],
            )
            features.append(parent)
        return BedTranscripts.add_introns(features)

    def _add_gene_model_track(self, canvas: Canvas, transcripts: Sequence[str]) -> None:
        """One row for the locus itself, labelled with the gene name."""
        if not self.gene_model:
            return
        features = self.gene_model_features(transcripts)
        if not features:
            return
        canvas.add(
            "GeneModelTrack", label=self.gene_name or "gene",
            reader=BedTranscripts(features=features),
            height=self.layout.gene_model,
            label_weight="bold",
        )

    def _add_transcript_tracks(self, canvas: Canvas, transcripts: Sequence[str]) -> None:
        bed = self.bed_frame()
        missing: List[str] = []
        # One track per isoform.
        for tid in transcripts:
            rows = bed[bed["name"] == tid] if not bed.empty else bed
            features = (_bed_rows_to_features(rows) if not rows.empty
                        else self._gtf_features({tid}))
            if not features:
                missing.append(tid)
                continue
            color = self.isoform_color(tid)
            reader = BedTranscripts(features=features)
            canvas.add(
                "TranscriptTrack", label=shorten_label(tid), reader=reader,
                height=self.layout.transcript,
                **{"style.exon.facecolor": color,
                   "style.CDS.facecolor": color,
                   "style.5UTR.facecolor": color,
                   "style.3UTR.facecolor": color},
            )

        if missing:
            self._say(f"  no model drawn for {', '.join(missing)}: "
                      "not in the BED12 or the annotation GTF")

    def _add_read_tracks(
        self, canvas: Canvas, transcripts: Sequence[str], conditions: Sequence[str],
        labels: Mapping[str, str], reads_by_transcript: Mapping[str, Set[str]],
        polya_fallback: Mapping[str, int], sort_style: str, max_reads: Optional[int],
    ) -> None:
        if not conditions:
            return

        groups: List[Tuple[str, Optional[Set[str]]]] = (
            [(tid, reads_by_transcript.get(tid, set())) for tid in transcripts]
            if transcripts else [("all reads", None)]
        )

        drawn = [(name, allowed) for name, allowed in groups
                 if allowed is None or allowed]

        for position, (name, allowed) in enumerate(drawn):
            color = self.isoform_color(name) if transcripts else self.theme.isoform_color(0)
            canvas.add(
                "SpacerTrack",
                height=self.layout.group_heading,
                heading=shorten_label(name),
                heading_color=color,
                rule_above=position > 0,
            )
            for condition in conditions:
                reader = LongReadBam(
                    pool=self.read_pool(condition),
                    allowed_read_names=allowed,
                    polya_lengths=polya_fallback,
                    strand_filter=self.strand,
                    polya_scale=self.polya_scale,
                )
                canvas.add(
                    "ReadTrack", label=labels[condition], reader=reader,
                    height=self.layout.read_stack,
                    sort_style=sort_style, max_reads=max_reads,
                    **{"style.exon.facecolor": color, "style.CDS.facecolor": color,
                       "style.5UTR.facecolor": color, "style.3UTR.facecolor": color},
                )

    @staticmethod
    def _normalise_mod_codes(mod_codes, verbose: bool) -> List[List[str]]:
        """Read ``mod_codes`` as groups, one track per group."""
        if not mod_codes:
            return []
        if all(isinstance(item, str) for item in mod_codes):
            groups = [[str(code) for code in mod_codes]]
        else:
            groups = [
                [str(code)] if isinstance(code, str) else [str(c) for c in code]
                for code in mod_codes
            ]

        capped = []
        for group in groups:
            if len(group) > MAX_OVERLAID_MODS and verbose:
                dropped = ", ".join(mod_label(c) for c in group[MAX_OVERLAID_MODS:])
                print(
                    f"  a modification track shows at most {MAX_OVERLAID_MODS} codes; "
                    f"dropped {dropped} (give them their own group for a second track)"
                )
            if group[:MAX_OVERLAID_MODS]:
                capped.append(group[:MAX_OVERLAID_MODS])
        return capped

    @property
    def mod_codes(self) -> List[str]:
        """Every modification code drawn, across all groups."""
        seen: List[str] = []
        for group in self.mod_groups:
            seen.extend(code for code in group if code not in seen)
        return seen

    def _add_modification_tracks(self, canvas: Canvas, conditions: Sequence[str],
                                 labels: Mapping[str, str]) -> None:
        """A lollipop track per group per condition, after the reads."""
        if not self.modkit_pileup or not self.mod_groups:
            return
        palette = self.theme.mod_palette(self.mod_codes)
        exons = self._mod_regions()
        first_track = True
        for group in self.mod_groups:
            first_of_group = True
            for condition in conditions:
                paths = self.modkit_pileup.get(condition)
                if not paths:
                    continue
                reader = ModPileupReader(
                    file_names=paths, codes=group,
                    min_coverage=self.min_mod_coverage,
                    min_fraction=self.min_mod_fraction,
                    strand_filter=self.strand,
                    regions=exons,
                )
                canvas.add(
                    "ModificationTrack", label=labels[condition], reader=reader,
                    height=self.layout.modification, colors=palette,
                    # Each group names its codes once, on its first track.
                    show_key=first_of_group,
                    rule_above=first_track,
                )
                first_of_group = False
                first_track = False

    def _coverage_read_names(self, transcripts: Sequence[str]) -> Optional[Set[str]]:
        """Reads the coverage track and its sashimi arcs are built from."""
        reads_by_transcript = self.gene_reads.reads_by_transcript
        if self.coverage_reads == COVERAGE_GENE or not reads_by_transcript:
            return self.gene_reads.read_names or None

        names: Set[str] = set()
        for transcript in transcripts:
            names.update(reads_by_transcript.get(transcript, ()))
        if not names:
            return self.gene_reads.read_names or None

        hidden = len(self.gene_reads.read_names) - len(names)
        if hidden > 0:
            self._say(
                f"  coverage from {len(names):,} reads on the {len(transcripts)} "
                f"isoform(s) drawn; {hidden:,} read(s) on other isoforms of this "
                "gene are excluded (coverage_reads='gene' includes them)"
            )
        return names

    def _mod_regions(self) -> Optional[List[Tuple[int, int]]]:
        """Spans the per-position tracks are allowed to draw in."""
        if self.mod_regions == MOD_REGIONS_WINDOW:
            return None
        return self.drawn_exons() or None

    def _add_mod_diff_tracks(self, canvas: Canvas) -> None:
        """One track per comparison: the change per site, signed."""
        if not self.mod_diff:
            return
        exons = self._mod_regions()
        first = True
        for label, path in self.mod_diff.items():
            reader = ModDiffReader(
                file_name=path,
                min_coverage=self.min_mod_coverage,
                max_p_value=self.max_diff_p_value,
                min_effect=self.min_diff_effect,
                balanced=self.balanced_diff,
                strand_filter=self.strand,
                regions=exons,
            )
            canvas.add(
                "ModDiffTrack", label=label, reader=reader,
                height=self.layout.modification_diff,
                show_key=first, rule_above=first and not self.modkit_pileup,
            )
            first = False

    def _add_coverage_tracks(self, canvas: Canvas, conditions: Sequence[str],
                             labels: Mapping[str, str], allowed: Optional[Set[str]]) -> None:
        read_filter = _name_filter(allowed) if allowed else None
        for position, condition in enumerate(conditions):
            canvas.add(
                "CoverageTrack", label=labels[condition],
                reader=_coverage_reader(self.bam_files[condition], self.strand, read_filter),
                height=self.layout.coverage,
                color=self.condition_colors.get(condition, self.theme.isoform_color(0)),
                rule_above=position == 0, scale_group=self.gene_name or "",
            )

    # ------------------------------------------------------------- plotting

    def plot(
        self,
        output: Optional[str] = None,
        formats: Optional[Sequence[str]] = None,
        region: Optional[Sequence] = None,
        padding: Union[int, str] = "auto",
        figsize: Optional[Tuple[float, float]] = None,
        width: Union[str, float, None] = None,
        sort_style: str = "waterfall",
        max_reads: Optional[int] = DEFAULT_MAX_READS,
        polya_scale: Optional[float] = None,
        share_coverage_scale: bool = True,
        highlight=None,
        highlight_style: str = "both",
        detail=None,
        detail_share: float = 0.4,
        detail_padding: int = 200,
        introns: str = INTRON_GENOMIC,
        intron_share: float = 0.3,
        dpi: int = 600,
        show: bool = False,
    ):
        """Draw the figure and optionally write it to *output*.

        Parameters
        ----------
        output : str, optional
            File to write, or a stem with no extension to write every format in *formats*.
        formats : sequence of str, optional
            Formats to write, e.g. ``("pdf", "png")``.
        region : tuple, optional
            Window to draw instead of the whole gene: the result of :meth:`zoom`, or
            ``(chrom, start, end)``.
        padding : int or ``"auto"``
            Base pairs of room at the 3' end.
        figsize : tuple, optional
            Figure size in inches, overriding the computed one.
        width : str or float, optional
            Printed width: ``"single"``, ``"onehalf"``, ``"double"``, ``"slide"``, or
            millimetres.
        sort_style : {"waterfall", "diamond"}
            Order of reads within a stack.
        max_reads : int or None
            Most reads drawn per stack; ``None`` draws all.
        polya_scale : float, optional
            Base pairs drawn per nucleotide of poly(A) tail, for this figure.
        share_coverage_scale : bool
            Put every condition's coverage on one depth axis.
        highlight : str, tuple, Highlight, or a list of them
            Regions to mark with a band across every track; see :meth:`highlight_region`.
        highlight_style : {"both", "box", "outline"}
            A faint box with a dotted edge, the box alone, or the edge alone.
        detail : str or tuple, optional
            Region to draw as a close-up beside the figure; see :meth:`detail_region`.
        detail_share : float
            Fraction of the data width the close-up takes.
        detail_padding : int
            Base pairs either side of the close-up region.
        introns : {"genomic", "scaled", "fixed"}
            Draw introns at true length, shrunk by one factor, or all at one width.
        intron_share : float
            With ``introns="scaled"``, the fraction of the width introns take.
        dpi : int
            Resolution of raster output.
        show : bool
            Show the figure in a window.
        """
        if width is not None:
            self.theme = self.theme.derive(width=width)
        if polya_scale is not None:
            self.polya_scale = float(polya_scale)
        self.theme.apply()
        if self.polya_scale != 1.0:
            self._say(
                f"poly(A) tails drawn at {self.polya_scale:g}x length"
            )

        if highlight_style not in HIGHLIGHT_STYLES:
            raise ValueError(
                f"unknown highlight_style {highlight_style!r}; expected one of "
                + ", ".join(repr(s) for s in HIGHLIGHT_STYLES)
            )
        if introns not in INTRON_MODES:
            raise ValueError(f"unknown introns {introns!r}; expected one of "
                             + ", ".join(repr(m) for m in INTRON_MODES))
        highlights = self._resolve_highlights(highlight)
        detail_window = None
        if detail is not None:
            detail_window, marker = self.detail_region(detail, padding=detail_padding)
            highlights = highlights + [marker]
            self._say(f"  detail panel: {marker.label or 'region'} "
                      f"{detail_window[0]}:{detail_window[1]:,}-{detail_window[2]:,}")

        window = (self._oriented(region) if region is not None
                  else self.display_region(padding))
        canvas = self.build(sort_style=sort_style, max_reads=max_reads, window=window)
        if any(h.label for h in highlights):
            canvas.prepend("HighlightLabelTrack", highlights=highlights)
        canvas.set_region(window)
        canvas.set_detail(detail_window, share=detail_share)
        canvas.set_intron_map(self._intron_map(window, introns, intron_share))

        if share_coverage_scale:
            self._share_coverage_scale(canvas)

        if figsize is None:
            figsize = self.layout.figure_size(sum(canvas.effective_heights()))

        figure = canvas.render(figsize=figsize)
        self.figure = figure
        draw_highlights(canvas, highlights, highlight_style, self.theme)

        for path in self._write(canvas, output, formats, dpi):
            self._say(f"wrote {path}")
        if show:
            plt.show()
        return figure

    @staticmethod
    def _write(canvas: Canvas, output: Optional[str],
               formats: Optional[Sequence[str]], dpi: int) -> List[str]:
        """Write the figure once per requested format; return the paths."""
        if not output:
            return []
        stem, extension = os.path.splitext(output)
        if formats:
            wanted = [f.lower().lstrip(".") for f in formats]
        elif extension:
            return [canvas.save(output, dpi=dpi)]
        else:
            wanted = list(DEFAULT_FORMATS)
        if extension and extension.lstrip(".").lower() not in wanted:
            # Keep an explicit extension alongside the listed formats.
            wanted.insert(0, extension.lstrip(".").lower())
        return [canvas.save(f"{stem}.{fmt}", dpi=dpi) for fmt in dict.fromkeys(wanted)]

    @staticmethod
    def _share_coverage_scale(canvas: Canvas) -> None:
        """One depth axis across each gene's coverage tracks, so heights compare."""
        groups: Dict[str, List] = {}
        for track in canvas.of_type("CoverageTrack"):
            groups.setdefault(track.config.get("scale_group", ""), []).append(track)
        for tracks in groups.values():
            if len(tracks) < 2:
                continue
            for track in tracks:
                track.load_region_data()
            high = max(track.preferred_ylim()[1] for track in tracks)
            for track in tracks:
                track.force_ylim(0.0, high)

    # ------------------------------------------------------------ tabular

    def polya_table(self, source: str = "auto") -> pd.DataFrame:
        """Per-read poly(A) tail lengths, for statistics rather than pictures.

        Parameters
        ----------
        source : {"auto", "annotation", "bam"}
            Where lengths come from.
        """
        if source not in ("auto", "annotation", "bam"):
            raise ValueError(f"unknown source {source!r}; use auto, annotation or bam")
        if source != "bam":
            table = self._polya_from_annotation()
            if table is not None:
                return table
            if source == "annotation":
                return pd.DataFrame(columns=POLYA_TABLE_COLUMNS)
        return self._polya_from_bams()

    def _polya_from_annotation(self) -> Optional[pd.DataFrame]:
        """Tail lengths as the read annotation recorded them, or ``None``."""
        frame = self.gene_reads.frame
        if frame.empty or "polya_length" not in frame.columns:
            return None

        table = frame.dropna(subset=["polya_length"]).copy()
        if table.empty:
            return pd.DataFrame(columns=POLYA_TABLE_COLUMNS)

        table["polya_length"] = table["polya_length"].astype(int)
        table = table[table["polya_length"] > 0]
        table["gene"] = self.gene_name

        if "sample" in table.columns and self.bam_files:
            mapping = map_samples_to_conditions(self.bam_files, self.gene_reads.samples)
            table["condition"] = table["sample"].map(mapping)
        else:
            table["sample"] = table.get("sample", "")
            table["condition"] = None

        for column in POLYA_TABLE_COLUMNS:
            if column not in table.columns:
                table[column] = None
        return table[POLYA_TABLE_COLUMNS].reset_index(drop=True)

    def _polya_from_bams(self) -> pd.DataFrame:
        """Tail lengths from the Dorado tags on the reads themselves."""
        read_to_transcript = self.gene_reads.read_index()
        allowed = self.gene_reads.read_names or None
        fallback = self.gene_reads.polya_lengths()

        chrom, start, end, region_strand = self.region
        rows: List[dict] = []
        for condition in self.bam_files:
            reader = LongReadBam(
                pool=self.read_pool(condition), allowed_read_names=allowed,
                polya_lengths=fallback, strand_filter=self.strand,
            )
            reader.chr_name, reader.region_start = chrom, start
            reader.region_end, reader.region_strand = end, region_strand
            for read in reader.iter_region_reads():
                length = reader.polya_length(read)
                if length <= 0:
                    continue
                rows.append({
                    "gene": self.gene_name,
                    "transcript_id": read_to_transcript.get(read_key(read.query_name), ""),
                    "condition": condition,
                    "sample": "",
                    "read_name": read.query_name,
                    "polya_length": length,
                })
        return pd.DataFrame(rows, columns=POLYA_TABLE_COLUMNS)


def _name_filter(allowed: Set[str]):
    """Read filter comparing read ids rather than whole, prefixed names."""
    keys = {read_key(name) for name in allowed}

    def keep(read) -> bool:
        return read_key(read.query_name) in keys

    return keep


def _coverage_reader(paths: Sequence[str], strand: str, read_filter):
    from .io.alignments import CoverageReader
    return CoverageReader(file_names=paths, strand_filter=strand, read_filter=read_filter)


def _coding_spans(record, part: str) -> List[Tuple[int, int]]:
    """Exonic spans of a BED12 record's ``5_utr``, ``3_utr`` or ``cds``."""
    start0, end = int(record["chromStart"]), int(record["chromEnd"])
    thick_start, thick_end = int(record["thickStart"]), int(record["thickEnd"])
    if thick_end <= thick_start:
        return []

    sizes = str(record.get("blockSizes", "")) if hasattr(record, "get") else ""
    offsets = str(record.get("blockStarts", "")) if hasattr(record, "get") else ""
    sizes = [int(v) for v in sizes.strip(",").split(",") if v.strip().isdigit()]
    offsets = [int(v) for v in offsets.strip(",").split(",") if v.strip().isdigit()]
    blocks = ([(start0 + o + 1, start0 + o + z) for o, z in zip(offsets, sizes)]
              if sizes and len(sizes) == len(offsets) else [(start0 + 1, end)])

    before = (start0 + 1, thick_start)  # upstream of the CDS in genome order
    after = (thick_end + 1, end)  # downstream of it
    plus = str(record["strand"]) != "-"
    low, high = {
        "cds": (thick_start + 1, thick_end),
        "5_utr": before if plus else after,
        "3_utr": after if plus else before,
    }[part]
    if high < low:
        return []
    return [(max(s, low), min(e, high)) for s, e in blocks
            if max(s, low) <= min(e, high)]


def _bed_rows_to_features(rows: pd.DataFrame):
    """Turn selected BED12 rows into transcript features without a temp file."""
    reader = BedTranscripts()
    features = []
    for record in rows.itertuples(index=False):
        fields = [str(getattr(record, name, "")) for name in
                  ("chrom", "chromStart", "chromEnd", "name", "score", "strand",
                   "thickStart", "thickEnd", "itemRgb", "blockCount",
                   "blockSizes", "blockStarts")]
        feature = reader._parse_line(fields, based0=True, region_type_name="mRNA")
        if feature is not None:
            features.append(feature)
    return BedTranscripts.add_introns(features)


def plot_gene(gene_name: str, bam_files: Optional[BamSpec] = None,
              isoforge: Optional[str] = None, output: Optional[str] = None, **options):
    """Build and draw a gene view in one call.

    *options* are the arguments of :class:`IsoTracks` and of :meth:`IsoTracks.plot`.
    """
    view_names = set(inspect.signature(IsoTracks.__init__).parameters) - {"self"}
    view_options = {name: options.pop(name) for name in list(options) if name in view_names}
    view = IsoTracks(gene_name=gene_name, bam_files=bam_files, isoforge=isoforge, **view_options)
    return view.plot(output=output, **options)
