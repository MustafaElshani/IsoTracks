"""The fusion figure: a schematic, then one panel per partner gene.

    from isotracks import FusionView

    view = FusionView(
        "RIMS2::ATP6V1C1",
        isoforge="results/project",
        bam_files={"Control": ["c1.bam"], "Treated": ["t1.bam"]},
        reference_gtf="gencode.v49.gtf",
    )
    view.plot(output="RIMS2_ATP6V1C1")
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from matplotlib import patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory, offset_copy

from . import theme as theme_module
from .annotation import GtfIndex, map_samples_to_conditions
from .browser import (
    CLUSTER_MATCH_WINDOW,
    DEFAULT_FORMATS,
    TAIL_CLEARANCE,
    BamSpec,
    add_site_tracks,
    normalise_bam_files,
)
from .canvas import Canvas, Panel
from .fusion import (
    BREAKPOINT_TOLERANCE,
    CONTEXT,
    END_QUANTILE,
    FusionCall,
    FusionRead,
    FusionReadReader,
    FusionTables,
    Partner,
    ReadGroup,
    _consensus,
    _ends,
    _transcript,
    collect_fusion_reads,
    fusion_read_features,
    group_reads,
    is_fusion_tables,
    junction_support,
    parse_fusion_line,
    read_ends,
    read_fusion_table,
    resolve_partners,
    _reaches_past,
)
from .highlight import Highlight
from .introns import INTRON_GENOMIC, INTRON_MODES, build_intron_map
from .isoforge import condition_of, find_outputs
from .io.clusters import ClusterSiteReader, SiteFrameReader
from .io.ends import EndPositionReader
from .io.features import BedTranscripts, Feature
from .io.longreads import polya_length_from_tags, read_key
from .layout import Layout
from .tracks.fusion_schematic import SchematicGene

#: Which transcripts the figure draws.
READS_ALL = "all"
READS_COMPLETE = "complete"
READ_SETS = (READS_ALL, READS_COMPLETE)
#: Room over a fusion row for the junction arcs and their counts, in layout units.
ARC_HEADROOM_UNITS = 1.3
#: Height of a junction arc's apex over the blocks it joins, in points.
ARC_RISE_PT = 5.5
#: Narrowest a partner's panel may be, as a share of the data width.
MIN_PANEL_SHARE = 0.14


class FusionView:
    """One fusion, drawn as a schematic over a panel per partner gene.

    Parameters
    ----------
    fusion : str or FusionCall
        The fusion: its genes (``"RIMS2::ATP6V1C1"``) or call ID (``"F000017"``) in
        the tables, a line of breakpoints, or a parsed call.
    fusion_table : str, optional
        IsoForge's fusion tables: their prefix, their directory or one of the
        ``*_isoforge_fusion*.parquet`` files.  Or a file of breakpoint lines.
    bam_files : mapping, optional
        ``{condition: bam}`` or ``{condition: [bams]}``.  With tables they group
        datasets into conditions and give the poly(A) tails; without tables the
        reads are found in them.
    isoforge : str, optional
        An IsoForge run: its output prefix, its directory or one of its files.
        Fills *fusion_table*, *tss*, *polya* and *bam_files*.
    conditions : mapping, optional
        ``{condition: [dataset, ...]}``, grouping the tables' datasets without BAMs.
        By default a dataset's condition is its name up to the first underscore.
    reference_gtf, annotation_gtf : str, optional
        Annotation for the schematic's whole genes; needed to place the partners
        when drawing from breakpoints.
    reads : {"all", "complete"}
        Draw every transcript of the call, or only those through every partner.
    introns : {"scaled", "fixed", "genomic"}
        Intron length within each panel.
    condition_labels : mapping, optional
        Display names for the conditions.
    tolerance : int
        Base pairs a read may miss a breakpoint by, when reads are found in BAMs.
    polya_scale : float
        Base pairs drawn per nucleotide of poly(A) tail.
    compact : bool
        Use the shorter track heights.  Ignored when *layout* is given.
    layout : Layout, optional
        Track heights.
    schematic : bool
        Draw the schematic above the panels.
    tss, polya : str, optional
        Clustered 5' and 3' end sites (``tss_clusters.bed``, ``tes_clusters.bed``).
    end_density : bool
        Draw the density of read 5' and 3' ends.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        fusion: Union[str, FusionCall],
        fusion_table: Optional[str] = None,
        bam_files: Optional[BamSpec] = None,
        isoforge: Optional[str] = None,
        conditions: Optional[Mapping[str, Sequence[str]]] = None,
        reference_gtf: Optional[str] = None,
        annotation_gtf: Optional[str] = None,
        reads: str = READS_ALL,
        introns: str = "scaled",
        condition_labels: Optional[Mapping[str, str]] = None,
        tolerance: int = BREAKPOINT_TOLERANCE,
        polya_scale: float = 1.0,
        compact: bool = True,
        layout: Optional[Layout] = None,
        schematic: bool = True,
        tss: Optional[str] = None,
        polya: Optional[str] = None,
        end_density: bool = True,
        verbose: bool = True,
    ):
        if reads not in READ_SETS:
            raise ValueError(f"unknown reads {reads!r}; expected one of {READ_SETS}")
        if introns not in INTRON_MODES:
            raise ValueError(f"unknown introns {introns!r}; expected one of {INTRON_MODES}")
        if isoforge:
            found = find_outputs(isoforge)
            bam_files = bam_files or found.get("bam_files")
            fusion_table = fusion_table or found.get("fusion_table")
            tss = tss or found.get("tss_clusters")
            polya = polya or found.get("polya_clusters")
        self.bam_files = normalise_bam_files(bam_files)
        self.conditions = {c: list(d) for c, d in (conditions or {}).items()}
        gtf = reference_gtf or annotation_gtf
        self.index = GtfIndex(gtf) if gtf else None

        self.tables: Optional[FusionTables] = None
        self.fusion_id: Optional[str] = None
        if is_fusion_tables(fusion_table if not isinstance(fusion, FusionCall) else None):
            self.tables = FusionTables(fusion_table)
            self.fusion_id, others = self.tables.find(str(fusion))
            self.call = self.tables.call(self.fusion_id)
            if others and verbose:
                print(f"{fusion}: drawing {self.fusion_id}, the best-supported of "
                      f"{len(others) + 1} calls with these genes; also {', '.join(others)}")
        else:
            self.call = self._load_call(fusion, fusion_table)
            if not self.bam_files:
                raise ValueError("without IsoForge's fusion tables, a fusion's reads are "
                                 "found in the BAMs: pass bam_files")
            if self.index is None:
                raise ValueError("without IsoForge's fusion tables, a GTF is needed to "
                                 "place the partner genes")

        self.read_set = reads
        self.introns = introns
        self.condition_labels = dict(condition_labels or {})
        self.tolerance = int(tolerance)
        self.polya_scale = float(polya_scale)
        self.layout = layout or (Layout.compact() if compact else Layout())
        self.schematic = schematic
        self.tss_clusters = tss
        self.polya_clusters = polya
        self.end_density = end_density
        self.transcript_names: List[str] = []
        self.verbose = verbose
        self.figure = None
        self.canvas: Optional[Canvas] = None

        self.partners: List[Partner] = []
        self.groups: List[ReadGroup] = []
        self.support: List[int] = []
        self.condition_order: List[str] = []

    @staticmethod
    def _load_call(fusion, fusion_table) -> FusionCall:
        if isinstance(fusion, FusionCall):
            return fusion
        if not isinstance(fusion, str):
            raise TypeError("fusion must be a name, an ID, a row of breakpoints, "
                            "or a FusionCall")
        if len(fusion.split()) >= 3:  # a whole row, pasted in
            return parse_fusion_line(fusion)
        if not fusion_table:
            raise ValueError(f"to draw {fusion!r} by name, pass the fusion_table it is in")
        return read_fusion_table(fusion_table, fusion)

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------- the data

    def prepare(self) -> None:
        """Place the partners, collect the reads and group them by transcript."""
        theme = theme_module.active()
        junctions = len(self.call.junctions)
        if self.tables is not None:
            self.partners = self.tables.partners(self.fusion_id)
            datasets = self.tables.datasets(self.fusion_id)
            condition_of = self._condition_map(datasets)
            reads = self.tables.fusion_reads(self.fusion_id, junctions, condition_of)
            self._add_tails(reads)
        else:
            self.partners = resolve_partners(self.call, self.index)
            found = collect_fusion_reads(self.partners, self.call.junctions,
                                         self.bam_files, self.tolerance)
            reads = [read for group in found.values() for read in group]
            self.condition_order = list(self.bam_files)
        if self.index is not None:
            for partner in self.partners:
                records = self.index.gene_records(partner.gene)
                partner.exons = [span for spans in records.exons.values() for span in spans]
        for partner, color in zip(self.partners, _cycle(theme.isoform_colors,
                                                         len(self.partners))):
            partner.color = color

        self.support = junction_support(reads, junctions)
        isoforms = self.tables.fusion_isoforms(self.fusion_id) if self.tables is not None else None
        groups = group_reads(self.partners, self.call.junctions, reads, self.tolerance,
                             isoforms=isoforms)
        if self.read_set == READS_COMPLETE:
            groups = [g for g in groups if g.complete]
        for group in groups:
            group.model = self._group_model(group)
        self.groups = groups
        self.transcript_names = self._transcript_labels()

        self._say(f"{self.call.name}"
                  + (f" ({self.fusion_id})" if self.fusion_id else "") + ": " + ", ".join(
                      f"{p.gene} {p.chrom} ({p.strand})" for p in self.partners))
        self._say("  reads across each junction: " + ", ".join(
            f"{self.call.genes[i]}>{self.call.genes[i + 1]} {n:,}"
            for i, n in enumerate(self.support)))
        for group, name in zip(self.groups, self.transcript_names):
            note = "" if group.complete else (
                f", {group.distinct:,} run past the {self._missing_joins(group)} join")
            placed = sum(r.fits is False for r in group.reads)
            if placed:
                note += f", {placed:,} placed with it as the transcript nearest theirs"
            described = f"{name} ({group.name})" if group.isoform_id else name
            self._say(f"  {described}: {len(group.reads):,} reads{note}")

    def _condition_map(self, datasets: Sequence[str]) -> Dict[str, str]:
        """Dataset to condition: as given, else from the BAMs, else from the dataset names."""
        if self.conditions:
            mapping = {d: c for c, members in self.conditions.items() for d in members}
            self.condition_order = list(self.conditions)
        elif self.bam_files:
            mapping = map_samples_to_conditions(self.bam_files, datasets)
            self.condition_order = list(self.bam_files)
        else:
            mapping = {d: condition_of(d) for d in datasets}
            self.condition_order = list(dict.fromkeys(mapping.values()))
        unplaced = [d for d in datasets if d not in mapping]
        if unplaced:
            self._say(f"  datasets in no condition, drawn on their own: {', '.join(unplaced)}")
            self.condition_order += unplaced
        return mapping

    def _add_tails(self, reads: Sequence[FusionRead]) -> None:
        """Poly(A) lengths from the BAMs' tags, off the last partner's pieces."""
        if not self.bam_files:
            return
        last = self.partners[-1]
        wanted = {read_key(r.name) for r in reads if last.index in r.pieces}
        if not wanted:
            return
        low = min(s for r in reads if last.index in r.pieces for s, _ in r.pieces[last.index])
        high = max(e for r in reads if last.index in r.pieces for _, e in r.pieces[last.index])
        tails: Dict[str, int] = {}
        for paths in self.bam_files.values():
            for path in paths:
                with pysam.AlignmentFile(path) as bam:
                    if last.chrom not in bam.references:
                        continue
                    for record in bam.fetch(last.chrom, max(low - 1, 0), high):
                        key = read_key(record.query_name)
                        if key in wanted and key not in tails:
                            tail = polya_length_from_tags(record)
                            if tail is not None and tail >= 0:
                                tails[key] = int(tail)
        for read in reads:
            read.polya_length = tails.get(read_key(read.name))

    def _group_model(self, group: ReadGroup) -> Dict[int, List[Tuple[int, int]]]:
        """Each partner's exons in this transcript: IsoForge's model, or its reads' consensus."""
        if group.isoform is not None and group.model:
            return group.model
        model = {}
        for index in group.parts:
            partner = self.partners[index]
            if group.complete and partner.model:
                model[index] = partner.model
                continue
            spans = _consensus([r.pieces[index] for r in group.reads if index in r.pieces])
            if spans:
                model[index] = spans
        return model

    def windows(self) -> List[Tuple[str, int, int, str]]:
        """Each partner's panel window: its transcripts, context and room for tails."""
        tail = max((r.polya_length or 0 for g in self.groups for r in g.reads), default=0)
        tail_room = int(tail * self.polya_scale) + TAIL_CLEARANCE if tail else 0
        last = len(self.partners) - 1
        windows = []
        for partner in self.partners:
            points = [partner.five_prime, partner.three_prime]
            for group in self.groups:
                for start, end in group.model.get(partner.index, []):
                    points += [start, end]
            starts = [min(s for s, _ in r.pieces[partner.index])
                      for g in self.groups for r in g.reads if partner.index in r.pieces]
            ends = [max(e for _, e in r.pieces[partner.index])
                    for g in self.groups for r in g.reads if partner.index in r.pieces]
            if starts:
                points += [int(np.quantile(starts, 1.0 - END_QUANTILE, method="lower")),
                           int(np.quantile(ends, END_QUANTILE, method="higher"))]
            low, high = min(points) - CONTEXT, max(points) + CONTEXT
            if partner.index == last:
                if partner.strand != "-":
                    high += tail_room
                else:
                    low -= tail_room
            windows.append((partner.chrom, max(1, low), high, partner.strand))
        return windows

    # ------------------------------------------------------------ the figure

    def build(self) -> Canvas:
        """The tracks and panels, before anything is drawn."""
        if not self.partners:
            self.prepare()
        theme = theme_module.active()
        conditions = self.condition_order or sorted({r.condition for g in self.groups
                                                     for r in g.reads})
        names = self.transcript_names or self._transcript_labels()
        fixed = ([self.condition_labels.get(c, c) for c in conditions] + ["read ends"]
                 + [n for n, g in zip(names, self.groups) if g.isoform_id])
        widest = max(len(t) for t in fixed)
        fusion_label = wrap_fusion_name(self.call.name, widest)
        row_labels = [n if g.isoform_id else wrap_fusion_name(n, widest)
                      for n, g in zip(names, self.groups)]
        lines = [line for text in row_labels for line in text.split("\n")]
        canvas = Canvas(gutter_in=self.layout.gutter_width_for(
            fixed + lines, bold=fusion_label.split("\n")))

        if self.schematic:
            canvas.add("FusionSchematicTrack", genes=self._schematic_genes(),
                       height=self.layout.fusion_schematic)
        canvas.add("HighlightLabelTrack", highlights=[
            Highlight([(p.start, p.end)], label=p.gene, label_color=p.color)
            for p in self.partners], rule_above=self.schematic)
        add_site_tracks(canvas, *self._sites(), theme, reader_for=SiteFrameReader)
        self._add_end_density_track(canvas)

        #: Rows joined across the gaps between panels: (track, {junction: keys of the rows crossing
        #: it}).
        self.joined_rows: List[Tuple[object, Dict[int, Optional[Set[str]]]]] = []
        crossed = sorted({j for group in self.groups for j in group.junctions})

        self.gene_track = canvas.add(
            "GeneModelTrack", label=fusion_label, label_weight="bold",
            label_lines=fusion_label.count("\n") + 1,
            reader=BedTranscripts(features=BedTranscripts.add_introns(
                [f for group in self.groups
                 for f in self._group_features(group, colour=False)])),
            # Headroom over the blocks for the junction arcs and their counts.
            height=self.layout.gene_model + ARC_HEADROOM_UNITS,
            y_margin=(0.02, ARC_HEADROOM_UNITS / self.layout.gene_model))
        self.joined_rows.append((self.gene_track, {j: None for j in crossed}))
        for group, label in zip(self.groups, row_labels):
            row = canvas.add("TranscriptTrack", label=label, height=self.layout.transcript,
                             label_lines=label.count("\n") + 1,
                             reader=BedTranscripts(features=BedTranscripts.add_introns(
                                 self._group_features(group))), hollow_utr=False)
            self.joined_rows.append((row, {j: None for j in group.junctions}))

        for group_index, (group, name) in enumerate(zip(self.groups, names)):
            heading, short = self._heading(group, name)
            canvas.add("SpacerTrack", height=self.layout.group_heading,
                       heading=heading, heading_short=short, rule_above=group_index > 0,
                       heading_panels="first")
            leading = {f"{r.sample}|{r.name}" for r in group.reads
                       if not group.complete
                       and _reaches_past(r, self.partners, self.call.junctions,
                                         self.tolerance)}
            for condition in conditions:
                members = [r for r in group.reads if r.condition == condition]
                features, rows = fusion_read_features(self.partners, members,
                                                      self.polya_scale, leading)
                stack = canvas.add(
                    "FusionReadTrack", label=self.condition_labels.get(condition, condition),
                    reader=FusionReadReader(features, rows), rows=len(members),
                    height=self.layout.read_stack if members else self.layout.transcript,
                    sublabel=f"n = {len(members):,}", label_lines=2,
                    **{"style.intron.facecolor": theme.intron_color,
                       "style.polyA.facecolor": theme.polya_color})
                self.joined_rows.append((stack, {
                    j: {f"{r.sample}|{r.name}" for r in members if j in r.junctions}
                    for j in group.junctions}))
        canvas.add("ScaleBarTrack", height=self.layout.scale_bar)
        canvas.add("CoordinateAxisTrack", height=self.layout.axis)

        windows = self.windows()
        models = [spans for g in self.groups for spans in g.model.values()]
        maps = [self._intron_map(partner, window, models)
                for partner, window in zip(self.partners, windows)]
        shares = self._shares(windows, maps)
        last = len(self.partners) - 1
        canvas.set_region(windows[0])
        canvas.panels[0] = Panel(region=canvas.region, share=shares[0], intron_map=maps[0],
                                 caption_text=_locus(windows[0]), scale_bar=last == 0)
        for partner, window, intron_map, share in zip(self.partners[1:], windows[1:],
                                                      maps[1:], shares[1:]):
            canvas.add_panel(Panel(region=window, share=share, intron_map=intron_map,
                                   caption_text=_locus(window),
                                   scale_bar=partner.index == last))
        canvas.of_type("CoordinateAxisTrack")[0].configure({"caption": _locus(windows[0])})
        self.canvas = canvas
        return canvas

    def _transcript_labels(self) -> List[str]:
        """Each transcript's label: IsoForge's ID, or the partners it joins."""
        labels = []
        for group in self.groups:
            if group.isoform_id:
                labels.append(group.isoform_id)
                continue
            same = [g for g in self.groups if g.parts == group.parts and not g.isoform_id]
            labels.append(group.name if len(same) < 2 else
                          f"{group.name} ({same.index(group) + 1} of {len(same)})")
        return labels

    def _heading(self, group: ReadGroup, name: Optional[str] = None) -> Tuple[str, str]:
        """The heading over a transcript's reads, and a shorter form for a narrow figure."""
        if name is None:
            name = self.transcript_names[self.groups.index(group)]
        title = f"{name} · {group.name}" if group.isoform_id else name
        if group.complete:
            return title, title
        counts = f"{group.distinct:,} of {len(group.reads):,}"
        return (f"{title} · {counts} reads run past the {self._missing_joins(group)} join",
                f"{title} · {counts} run past the join")

    def _missing_joins(self, group: ReadGroup) -> str:
        """The partners the whole fusion joins where this transcript stops short."""
        return " or ".join(self.partners[i].gene
                           for i in (group.parts[0] - 1, group.parts[-1] + 1)
                           if 0 <= i < len(self.partners))

    def _group_features(self, group: ReadGroup, colour: bool = True) -> List[Feature]:
        """One transcript as a feature per partner, in the partner's colour or uncoloured."""
        return [_transcript(self.partners[i].chrom, self.partners[i].strand, spans,
                            self.partners[i].gene,
                            self.partners[i].color if colour else None)
                for i, spans in sorted(group.model.items())]

    def _sites(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """poly(A) and TSS clusters at the ends of the transcripts drawn."""
        found = []
        for path, three_prime, motif in ((self.polya_clusters, True, True),
                                         (self.tss_clusters, False, False)):
            frames = []
            if path and os.path.exists(path) and self.groups:
                for partner, window in zip(self.partners, self.windows()):
                    # The transcripts that start (or end) in this partner.
                    ends = [_ends(partner, group.model[partner.index])[int(three_prime)]
                            for group in self.groups
                            if group.model.get(partner.index)
                            and partner.index == group.parts[-1 if three_prime else 0]]
                    if not ends:
                        continue
                    sites = ClusterSiteReader(path, require_motif=motif).read_region(window)
                    sites = sites[sites["strand"] == partner.strand]
                    near = sites["site"].apply(lambda site: any(
                        abs(int(site) - end) <= CLUSTER_MATCH_WINDOW for end in ends))
                    frames.append(sites[near])
            found.append(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        return found[0], found[1]

    def _add_end_density_track(self, canvas: Canvas) -> None:
        """Where the fusion's reads start and stop, pooled across conditions."""
        reads = [r for group in self.groups for r in group.reads]
        if not self.end_density or not reads:
            return
        reader = EndPositionReader(*read_ends(self.partners, reads))
        peak = max(reader.read_region(window).peak for window in self.windows())
        canvas.add("EndDensityTrack", label="read ends", reader=reader,
                   height=self.layout.end_density, fix_ylim=(0.0, max(peak, 1e-9)))

    def _schematic_genes(self) -> List[SchematicGene]:
        """The partners as the schematic draws them."""
        whole = next((g for g in self.groups if g.complete), None)
        junctions = self.call.junctions
        last = len(self.partners) - 1
        genes = []
        for partner in self.partners:
            i = partner.index
            exons = (self.index.gene_records(partner.gene).representative()[1]
                     if self.index is not None else [])
            acceptor = junctions[i - 1].acceptor if i > 0 else None
            donor = junctions[i].donor if i < last else None
            kept = ((whole.model.get(i) if whole is not None else None)
                    or partner.model or _clip_to_kept(exons, partner.strand, acceptor, donor)
                    or [(partner.start, partner.end)])
            genes.append(SchematicGene(
                name=partner.gene, chrom=partner.chrom, strand=partner.strand,
                color=partner.color, exons=list(exons), kept=list(kept),
                acceptor=acceptor, donor=donor))
        return genes

    def _intron_map(self, partner: Partner, window, models):
        if self.introns == INTRON_GENOMIC:
            return None
        exons = [(s, e) for s, e in partner.exons if e >= window[1] and s <= window[2]]
        exons += [span for spans in models for span in spans
                  if span[1] >= window[1] and span[0] <= window[2]]
        return build_intron_map(exons, window, self.introns) if exons else None

    @staticmethod
    def _shares(windows, maps) -> List[float]:
        """Panel widths in proportion to drawn length, none narrower than a floor."""
        drawn = []
        for window, intron_map in zip(windows, maps):
            low, high = float(window[1]), float(window[2])
            drawn.append(float(intron_map.forward(high) - intron_map.forward(low))
                         if intron_map is not None else high - low)
        total = sum(drawn) or 1.0
        shares = [max(d / total, MIN_PANEL_SHARE) for d in drawn]
        scale = sum(shares)
        return [share / scale for share in shares]

    def plot(self, output: Optional[str] = None,
             formats: Optional[Sequence[str]] = None, dpi: int = 600,
             width: Union[str, float, None] = None, show: bool = False):
        """Draw the fusion and optionally write it to *output*."""
        theme = theme_module.active()
        if width is not None:
            theme = theme.derive(width=width)
        theme.apply()
        canvas = self.build()
        figsize = self.layout.figure_size(sum(canvas.effective_heights()))
        figure = canvas.render(figsize=figsize)
        self.figure = figure
        self._draw_breakpoints(canvas, theme)
        self._draw_joins(canvas, theme)
        self._draw_junction_arcs(canvas, theme, self.gene_track)
        for path in self._write(canvas, output, formats, dpi):
            self._say(f"wrote {path}")
        if show:
            plt.show()
        return figure

    # ---------------------------------------------------- breakpoints, arcs

    @staticmethod
    def _panel_track(index: int, track):
        """The track, or its copy, that drew panel *index*."""
        return track if index == 0 else track.panel_copies[index - 1]

    @classmethod
    def _panel_axes(cls, index: int, track):
        return cls._panel_track(index, track).ax

    def _edges(self, junction_index: int) -> Tuple[float, float]:
        """Screen edges of a junction: the donor's 3' side, the acceptor's 5'."""
        junction = self.call.junctions[junction_index]
        upstream = self.partners[junction_index]
        downstream = self.partners[junction_index + 1]
        donor = junction.donor + (0.5 if upstream.strand != "-" else -0.5)
        acceptor = junction.acceptor - (0.5 if downstream.strand != "-" else -0.5)
        return donor, acceptor

    def _draw_breakpoints(self, canvas: Canvas, theme) -> None:
        """A dashed line through every track at each breakpoint of the whole fusion."""
        figure = canvas.fig
        runs: List[list] = [[]]
        for track in canvas.tracks:
            kind = type(track).__name__
            if kind == "SpacerTrack" and track.config.get("heading"):
                runs.append([])
            elif kind not in ("HighlightLabelTrack", "SpacerTrack", "ScaleBarTrack",
                              "CoordinateAxisTrack") and not track.SPANS_PANELS:
                runs[-1].append(track)
        for junction_index in range(len(self.call.junctions)):
            donor, acceptor = self._edges(junction_index)
            for panel_index, x in ((junction_index, donor), (junction_index + 1, acceptor)):
                for run in filter(None, runs):
                    axes = [self._panel_axes(panel_index, t) for t in run]
                    top = max(ax.get_position().y1 for ax in axes)
                    bottom = min(ax.get_position().y0 for ax in axes)
                    transform = blended_transform_factory(axes[0].transData,
                                                          figure.transFigure)
                    figure.add_artist(Line2D(
                        [x, x], [bottom, top], transform=transform,
                        color=theme.ink["axis"], linewidth=0.6 * theme.stroke,
                        linestyle=(0, (2.0, 1.6)), zorder=6, gid="fusion-breakpoint"))

    def spliced(self, junction_index: int) -> bool:
        """Whether a junction is on exon boundaries at both ends."""
        return self.call.junctions[junction_index].splice_sites == "both"

    def _draw_joins(self, canvas: Canvas, theme) -> None:
        """Join every transcript and read across the gap at each junction it crosses."""
        figure = canvas.fig
        to_figure = figure.transFigure.inverted()
        bands, colours, dots = [], [], []
        for track, crossings in self.joined_rows:
            for junction_index, keys in crossings.items():
                if keys is not None and not keys:
                    continue
                left = self._panel_track(junction_index, track)
                right = self._panel_track(junction_index + 1, track)
                if left.region_data is None or right.region_data is None:
                    continue
                ends = {_row_key(f): f for f in right.region_data}
                upstream = self.partners[junction_index]
                downstream = self.partners[junction_index + 1]
                for feature in left.region_data:
                    if keys is not None and _row_key(feature) not in keys:
                        continue
                    other = (ends.get(_row_key(feature)) if keys is not None
                             else next(iter(ends.values()), None))
                    if other is None:
                        continue
                    if keys is None:  # a model ends at the junction
                        start, end = self._edges(junction_index)
                    else:  # a read where it was cut
                        start = _edge(feature, upstream, three_prime=True)
                        end = _edge(other, downstream, three_prime=False)
                    bottom, height, colour = left.intron_band(feature)
                    (x0, y0), (_, y1) = to_figure.transform(left.ax.transData.transform(
                        [(start, bottom), (start, bottom + height)]))
                    x1 = to_figure.transform(right.ax.transData.transform((end, bottom)))[0]
                    low, high = sorted((y0, y1))
                    if self.spliced(junction_index):
                        bands.append([(x0, low), (x1, low), (x1, high), (x0, high)])
                        colours.append(colour)
                    else:
                        dots.append([(x0, (low + high) / 2), (x1, (low + high) / 2)])
                    if keys is None:
                        break  # one join per model row
        if bands:
            figure.add_artist(PolyCollection(
                bands, facecolors=colours, edgecolors="none", transform=figure.transFigure,
                zorder=1, gid="fusion-join-intron"))
        if dots:
            figure.add_artist(LineCollection(
                dots, colors=theme.ink["muted"], linewidths=0.7 * theme.stroke,
                linestyles=(0, (1.0, 1.4)), transform=figure.transFigure, zorder=1,
                gid="fusion-join-direct"))

    def _draw_junction_arcs(self, canvas: Canvas, theme, track) -> None:
        """An arc over each junction on the fusion gene row, with its read count."""
        figure = canvas.fig
        to_figure = figure.transFigure.inverted()
        rise = ARC_RISE_PT / 72.0 / figure.get_figheight()
        reads = [r for group in self.groups for r in group.reads]
        for junction_index in range(len(self.call.junctions)):
            count = sum(junction_index in r.junctions for r in reads)
            if not count:
                continue
            donor, acceptor = self._edges(junction_index)
            left = self._panel_track(junction_index, track)
            right = self._panel_track(junction_index + 1, track)
            start = to_figure.transform(left.ax.transData.transform(
                (donor, left.data_ylim[1])))
            end = to_figure.transform(right.ax.transData.transform(
                (acceptor, right.data_ylim[1])))
            apex_y = max(start[1], end[1]) + rise
            middle_x = (start[0] + end[0]) / 2.0
            # The control point sits at twice the apex height.
            path = Path([start, (middle_x, 2 * apex_y - (start[1] + end[1]) / 2), end],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            figure.add_artist(PathPatch(
                path, transform=figure.transFigure, facecolor="none",
                edgecolor=theme.ink["axis"], linewidth=0.7 * theme.stroke, zorder=7,
                gid="fusion-junction"))
            text = figure.text(
                middle_x, apex_y, f"{count:,}", transform=offset_copy(
                    figure.transFigure, fig=figure, y=1.0, units="points"),
                ha="center", va="bottom", color=theme.ink["text"],
                fontsize=theme.fonts["annot"], zorder=8, gid="fusion-junction-count")
            if not theme.transparent:
                text.set_path_effects([patheffects.withStroke(linewidth=2.2,
                                                              foreground="white"),
                                       patheffects.Normal()])

    @staticmethod
    def _write(canvas: Canvas, output, formats, dpi) -> List[str]:
        if not output:
            return []
        stem, extension = os.path.splitext(output)
        if formats:
            wanted = [f.lower().lstrip(".") for f in formats]
        elif extension:
            return [canvas.save(output, dpi=dpi)]
        else:
            wanted = list(DEFAULT_FORMATS)
        return [canvas.save(f"{stem}.{fmt}", dpi=dpi) for fmt in dict.fromkeys(wanted)]


def _cycle(colors: Sequence[str], count: int) -> List[str]:
    return [colors[i % len(colors)] for i in range(count)]


def _locus(window) -> str:
    return f"{window[0]}  {window[3]}"


def _row_key(feature) -> str:
    """What matches a row across panels: its read, or the model it belongs to."""
    return feature.read_name or feature.id


def _edge(feature, partner: Partner, three_prime: bool) -> float:
    """The drawn edge of a piece at its 5' or 3' end, on its partner's strand."""
    forward = partner.strand != "-"
    if three_prime == forward:
        return feature.end + 0.5
    return feature.start - 0.5


def wrap_fusion_name(name: str, width: int) -> str:
    """A fusion's name on as many lines as fit *width* characters, broken after a ``::``."""
    genes, _, number = name.partition(" (")
    genes = genes.split("::")
    parts = [g + "::" for g in genes[:-1]] + genes[-1:] + ([f"({number}"] if number else [])
    lines: List[str] = []
    for part in parts:
        joiner = " " if part.startswith("(") else ""
        if lines and len(lines[-1]) + len(joiner) + len(part) <= width:
            lines[-1] += joiner + part
        else:
            lines.append(part)
    return "\n".join(lines)


def _clip_to_kept(exons: Sequence[Tuple[int, int]], strand: str,
                  acceptor: Optional[int], donor: Optional[int]) -> List[Tuple[int, int]]:
    """A gene's exons cut to the part a fusion keeps."""
    low, high = (acceptor, donor) if strand != "-" else (donor, acceptor)
    low = -np.inf if low is None else low
    high = np.inf if high is None else high
    return [(int(max(s, low)), int(min(e, high))) for s, e in exons if s <= high and e >= low]
