"""Fusion data: calls, partner genes, reads and transcripts.

Read from IsoForge's fusion tables (:class:`FusionTables`), or from a line of
breakpoints with the reads found in the BAMs:

    RIMS2::ATP6V1C1   chr8::chr8   chr8:103501062>chr8:103062955   both

A junction is ``donor>acceptor``: the last base kept from the upstream partner
and the first base kept from the downstream one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pysam

from .io._intervals import merge_intervals
from .io.features import FEATURE_HEADER, BedTranscripts, Feature, FeatureArray
from .io.longreads import polya_length_from_tags

#: Fields of a breakpoint line, by position.
FUSION_COLUMNS = ("name", "chromosomes", "breakpoints", "splice_sites",
                  "reads", "unused", "filter")

#: Base pairs a read's piece may miss a breakpoint by and still count as ending there.
BREAKPOINT_TOLERANCE = 15
#: Context drawn either side of each partner's kept segment, in base pairs.
CONTEXT = 150
#: Gaps inside one alignment shorter than this are closed as deletions.
MIN_INTRON = 30


# ------------------------------------------------------------------ parsing

@dataclass
class Junction:
    """Where one partner hands over to the next."""

    donor_chrom: str
    donor: int  # last base kept from the upstream partner
    acceptor_chrom: str
    acceptor: int  # first base kept from the downstream partner
    splice_sites: str = ""


@dataclass
class FusionCall:
    """One row of IsoForge's fusion table."""

    name: str
    genes: List[str]
    chromosomes: List[str]
    junctions: List[Junction]
    reads: Optional[int] = None
    status: str = ""
    fields: Dict[str, str] = field(default_factory=dict)


def parse_breakpoints(text: str) -> List[Junction]:
    """``chr9:107283561>chr8:103500557;...`` into junctions."""
    junctions = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        try:
            donor, acceptor = part.split(">")
            donor_chrom, donor_pos = donor.rsplit(":", 1)
            acceptor_chrom, acceptor_pos = acceptor.rsplit(":", 1)
            junctions.append(Junction(donor_chrom, int(donor_pos.replace(",", "")),
                                      acceptor_chrom, int(acceptor_pos.replace(",", ""))))
        except ValueError:
            raise ValueError(f"bad breakpoint {part!r}; expected chrA:123>chrB:456") from None
    if not junctions:
        raise ValueError(f"no breakpoints in {text!r}")
    return junctions


def parse_fusion_line(line: str) -> FusionCall:
    """One fusion-table row, split on whitespace and read by position."""
    values = line.split()
    if len(values) < 3:
        raise ValueError(f"a fusion row needs at least name, chromosomes and "
                         f"breakpoints; got {line!r}")
    row = dict(zip(FUSION_COLUMNS, values))
    genes = row["name"].split("::")
    chromosomes = row["chromosomes"].split("::")
    junctions = parse_breakpoints(row["breakpoints"])

    if not len(genes) == len(chromosomes) == len(junctions) + 1:
        raise ValueError(
            f"{row['name']}: {len(genes)} genes and {len(chromosomes)} chromosomes "
            f"need {len(genes) - 1} junctions, found {len(junctions)}")
    for index, junction in enumerate(junctions):
        if (junction.donor_chrom, junction.acceptor_chrom) != (
                chromosomes[index], chromosomes[index + 1]):
            raise ValueError(
                f"{row['name']}: junction {index + 1} joins {junction.donor_chrom} to "
                f"{junction.acceptor_chrom}, but the partners are on "
                f"{chromosomes[index]} and {chromosomes[index + 1]}")

    sites = row.get("splice_sites", "").split(";")
    for junction, site in zip(junctions, sites):
        junction.splice_sites = site
    reads = row.get("reads")
    return FusionCall(
        name=row["name"], genes=genes, chromosomes=chromosomes, junctions=junctions,
        reads=int(reads) if reads and reads.isdigit() else None,
        status=row.get("filter", ""), fields=row,
    )


def read_fusion_table(path: str, name: str) -> FusionCall:
    """The row for fusion *name* in an IsoForge fusion table."""
    found = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            first = line.split()[0]
            if first == name:
                found.append(parse_fusion_line(line))
    if not found:
        raise ValueError(f"fusion {name!r} is not in {os.path.basename(path)}")
    return found[0]


# ----------------------------------------------------------------- partners

@dataclass
class Partner:
    """One gene of a fusion, and the part of it the fusion keeps."""

    index: int
    gene: str
    chrom: str
    strand: str
    five_prime: int  # first kept base, in transcript order
    three_prime: int  # last kept base
    color: str = ""
    exons: List[Tuple[int, int]] = field(default_factory=list)  # annotated
    #: The whole gene's span, when known.
    gene_start: Optional[int] = None
    gene_end: Optional[int] = None
    #: This part of the fusion transcript, as IsoForge called it.
    model: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def start(self) -> int:
        return min(self.five_prime, self.three_prime)

    @property
    def end(self) -> int:
        return max(self.five_prime, self.three_prime)

    def is_five_prime_of(self, a: int, b: int) -> bool:
        """Whether *a* comes before *b* in this partner's transcript."""
        return a < b if self.strand == "+" else a > b


def resolve_partners(call: FusionCall, index) -> List[Partner]:
    """Place each partner from the GTF and cut out the part the fusion keeps."""
    partners = []
    last = len(call.genes) - 1
    for position, gene in enumerate(call.genes):
        records = index.gene_records(gene)
        extent = records.transcript_extent() or records.gene
        if extent is None:
            raise ValueError(f"fusion partner {gene} is not in the GTF")
        if extent.chromosome != call.chromosomes[position]:
            raise ValueError(f"{gene} is on {extent.chromosome} in the GTF, but the "
                             f"fusion places it on {call.chromosomes[position]}")
        plus = extent.strand != "-"
        five = (call.junctions[position - 1].acceptor if position
                else (extent.start if plus else extent.end))
        three = (call.junctions[position].donor if position < last
                 else (extent.end if plus else extent.start))
        partner = Partner(position, gene, extent.chromosome, extent.strand, five, three,
                          exons=merge_intervals(
                              span for spans in records.exons.values() for span in spans))
        if partner.is_five_prime_of(three, five):
            raise ValueError(
                f"{gene} is on the {extent.strand} strand, but the fusion keeps it from "
                f"{five:,} to {three:,}, which runs 3' to 5'; the breakpoints do not "
                f"fit the gene's orientation")
        partners.append(partner)
    return partners


# -------------------------------------------------------------------- reads

@dataclass
class FusionRead:
    """One read through the fusion: its pieces, partner by partner."""

    name: str
    condition: str
    sample: str
    pieces: Dict[int, List[Tuple[int, int]]]  # partner index -> blocks
    junctions: Set[int]  # junctions it spans
    polya_length: Optional[int] = None
    #: IsoForge's transcript this read belongs to, and whether the read fits it.
    isoform: Optional[int] = None
    fits: Optional[bool] = None

    @property
    def parts(self) -> Tuple[int, ...]:
        """The partners this read has a piece on, in transcript order."""
        return tuple(sorted(self.pieces))


def _blocks(read: pysam.AlignedSegment) -> List[Tuple[int, int]]:
    """Aligned blocks, 1-based inclusive, with deletions closed up."""
    merged: List[List[int]] = []
    for start, end in read.get_blocks():
        if merged and start - merged[-1][1] < MIN_INTRON:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start + 1, end) for start, end in merged]


def _ends(partner: Partner, blocks) -> Tuple[int, int]:
    """(5' end, 3' end) of a piece, in the partner's transcript order."""
    low, high = min(s for s, _ in blocks), max(e for _, e in blocks)
    return (low, high) if partner.strand != "-" else (high, low)


def read_ends(partners: Sequence[Partner], reads: Sequence["FusionRead"]
              ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Every read's 5' end and 3' end, by chromosome."""
    fives: Dict[str, List[int]] = {}
    threes: Dict[str, List[int]] = {}
    for read in reads:
        first, last = partners[read.parts[0]], partners[read.parts[-1]]
        fives.setdefault(first.chrom, []).append(_ends(first, read.pieces[first.index])[0])
        threes.setdefault(last.chrom, []).append(_ends(last, read.pieces[last.index])[1])
    return fives, threes


def collect_fusion_reads(partners: Sequence[Partner], junctions: Sequence[Junction],
                         bam_files: Mapping[str, Sequence[str]],
                         tolerance: int = BREAKPOINT_TOLERANCE,
                         reach: int = 20_000) -> Dict[str, List[FusionRead]]:
    """Reads across at least one junction, per condition."""
    last = len(partners) - 1
    found: Dict[str, List[FusionRead]] = {}
    for condition, paths in bam_files.items():
        reads: List[FusionRead] = []
        for path in paths:
            pieces: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
            tails: Dict[str, int] = {}
            with pysam.AlignmentFile(path) as bam:
                for partner in partners:
                    if partner.chrom not in bam.references:
                        continue
                    low = partner.start - (reach if partner.index in (0, last) else 0)
                    high = partner.end + (reach if partner.index in (0, last) else 0)
                    for read in bam.fetch(partner.chrom, max(low - tolerance - 1, 0),
                                          high + tolerance):
                        if read.is_unmapped or read.is_secondary:
                            continue
                        if ("-" if read.is_reverse else "+") != partner.strand:
                            continue
                        blocks = _blocks(read)
                        if not blocks:
                            continue
                        pieces.setdefault(read.query_name, {}).setdefault(
                            partner.index, []).extend(blocks)
                        tail = polya_length_from_tags(read)
                        if tail is not None and tail >= 0:
                            tails[read.query_name] = int(tail)

            for name, by_partner in pieces.items():
                spanned = set()
                for index, junction in enumerate(junctions):
                    upstream, downstream = by_partner.get(index), by_partner.get(index + 1)
                    if not upstream or not downstream:
                        continue
                    three = _ends(partners[index], upstream)[1]
                    five = _ends(partners[index + 1], downstream)[0]
                    if (abs(three - junction.donor) <= tolerance
                            and abs(five - junction.acceptor) <= tolerance):
                        spanned.add(index)
                if spanned:
                    reads.append(FusionRead(
                        name=name, condition=condition, sample=os.path.basename(path),
                        pieces={k: merge_intervals(v) for k, v in by_partner.items()},
                        junctions=spanned, polya_length=tails.get(name)))
        found[condition] = reads
    return found


#: Share of reads whose outer ends the panels are sized to hold.
END_QUANTILE = 0.9


def complete_reads(reads: Sequence[FusionRead], junctions: int) -> List[FusionRead]:
    """Reads that cross every junction."""
    return [read for read in reads if len(read.junctions) == junctions]


def junction_support(reads: Sequence[FusionRead], junctions: int) -> List[int]:
    """How many reads cross each junction, whatever else they cross."""
    return [sum(index in read.junctions for read in reads) for index in range(junctions)]


def observed_ends(partners: Sequence[Partner], reads: Sequence[FusionRead],
                  quantile: float = END_QUANTILE) -> None:
    """Narrow the outer ends of the fusion to what its reads show."""
    if not reads:
        return
    first, last = partners[0], partners[-1]
    starts = [_ends(first, r.pieces[0])[0] for r in reads if 0 in r.pieces]
    stops = [_ends(last, r.pieces[last.index])[1] for r in reads if last.index in r.pieces]
    if starts:
        upstream = np.quantile(starts, 1.0 - quantile, method="lower") \
            if first.strand != "-" else np.quantile(starts, quantile, method="higher")
        first.five_prime = int(upstream)
    if stops:
        downstream = np.quantile(stops, quantile, method="higher") \
            if last.strand != "-" else np.quantile(stops, 1.0 - quantile, method="lower")
        last.three_prime = int(downstream)


# ----------------------------------------------------------------- features

def _transcript(chrom, strand, spans, feature_id, color=None) -> Feature:
    spans = sorted(spans)
    parm = {"facecolor": color} if color else None
    return Feature(spans[0][0], spans[-1][1], chrom, strand, feature_id, 1, "mRNA",
                   children=[Feature(s, e, chrom, strand, "", 1, "exon", parm=parm)
                             for s, e in spans])


def fusion_model(partners: Sequence[Partner],
                 reads: Sequence[FusionRead]) -> List[Feature]:
    """The fusion transcript, one feature per partner in the partner's colour."""
    features = []
    for partner in partners:
        spans = _consensus([r.pieces[partner.index] for r in reads
                            if partner.index in r.pieces])
        if not spans:
            spans = [(max(s, partner.start), min(e, partner.end)) for s, e in partner.exons
                     if e >= partner.start and s <= partner.end]
        if spans:
            features.append(_transcript(partner.chrom, partner.strand, spans,
                                        partner.gene, partner.color))
    return BedTranscripts.add_introns(features)


def _consensus(pieces: Sequence[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    if not pieces:
        return []
    low = min(s for blocks in pieces for s, _ in blocks)
    high = max(e for blocks in pieces for _, e in blocks)
    depth = np.zeros(high - low + 2, dtype=np.int32)
    for blocks in pieces:
        for start, end in blocks:
            depth[start - low] += 1
            depth[end - low + 1] -= 1
    covered = np.cumsum(depth)[:-1] >= max(1, int(np.ceil(len(pieces) / 2)))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], covered.astype(int), [0]))))
    return [(low + int(a), low + int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def fusion_read_features(partners: Sequence[Partner], reads: Sequence[FusionRead],
                         polya_scale: float = 1.0,
                         leading: Optional[Set[str]] = None
                         ) -> Tuple[List[Feature], Dict[str, int]]:
    """Every piece of every read as a feature, and each read's row."""
    last = partners[-1].index
    leading = leading or set()
    ordered = sorted(reads, key=lambda r: (f"{r.sample}|{r.name}" not in leading,
                                           -(r.polya_length or 0), r.sample, r.name))
    rows = {f"{r.sample}|{r.name}": row for row, r in enumerate(ordered)}
    features = []
    for read in ordered:
        key = f"{read.sample}|{read.name}"
        for index, blocks in sorted(read.pieces.items()):
            partner = partners[index]
            feature = _transcript(partner.chrom, partner.strand, blocks, key, partner.color)
            feature.read_name = key
            tail = read.polya_length or 0
            feature.polya_length = tail
            feature.read_type = "polya" if tail > 0 else "untailed"
            visual = int(round(tail * polya_scale))
            if index == last and visual > 0:
                if partner.strand != "-":
                    feature.children.append(Feature(feature.end + 1, feature.end + visual,
                                                    partner.chrom, partner.strand, "", 1,
                                                    "polyA"))
                    feature.end += visual
                else:
                    feature.children.insert(0, Feature(feature.start - visual,
                                                       feature.start - 1, partner.chrom,
                                                       partner.strand, "", 1, "polyA"))
                    feature.start -= visual
            features.append(feature)
    return BedTranscripts.add_introns(features), rows


class FusionReadReader:
    """The pieces of one condition's fusion reads, served a window at a time."""

    def __init__(self, features: Sequence[Feature], rows: Mapping[str, int]):
        self.features = list(features)
        self.rows = dict(rows)
        self.region_data: Optional[FeatureArray] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region, select_features=None) -> None:
        chrom, start, end, _strand = region
        low, high = sorted((int(start), int(end)))
        kept = [f for f in self.features
                if f.chr_name == chrom and f.end >= low and f.start <= high]
        array = FeatureArray(features=kept, header=list(FEATURE_HEADER))
        array.df["polya_length"] = [f.polya_length for f in kept]
        array.df["read_name"] = [f.read_name for f in kept]
        array.df["read_type"] = [f.read_type for f in kept]
        array.df["order"] = [self.rows[f.read_name] for f in kept]
        self.region_data = array


# ------------------------------------------------------ IsoForge's tables

#: IsoForge's fusion tables, by suffix.
TABLE_SUFFIXES = {
    "calls": "_isoforge_fusions.parquet",
    "parts": "_isoforge_fusion_parts.parquet",
    "reads": "_isoforge_fusion_reads.parquet",
}
#: The transcripts table: each call's transcripts and their IDs.
ISOFORM_SUFFIX = "_isoforge_fusion_isoforms.parquet"


def _table_paths(source: str) -> Dict[str, str]:
    """The three table paths from a prefix, a directory, or any one of them."""
    if os.path.isdir(source):
        found = [name for name in os.listdir(source)
                 if name.endswith(TABLE_SUFFIXES["calls"])]
        if len(found) != 1:
            raise ValueError(f"expected one *{TABLE_SUFFIXES['calls']} in {source}, "
                             f"found {len(found)}")
        source = os.path.join(source, found[0])
    prefix = source
    for suffix in (*TABLE_SUFFIXES.values(), ISOFORM_SUFFIX):
        if source.endswith(suffix):
            prefix = source[: -len(suffix)]
    paths = {kind: prefix + suffix for kind, suffix in TABLE_SUFFIXES.items()}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise ValueError("fusion table(s) not found:\n  " + "\n  ".join(missing))
    if os.path.exists(prefix + ISOFORM_SUFFIX):
        paths["isoforms"] = prefix + ISOFORM_SUFFIX
    return paths


def is_fusion_tables(source: Optional[str]) -> bool:
    """Whether *source* names IsoForge's Parquet tables rather than a TSV row."""
    if not source:
        return False
    try:
        _table_paths(source)
        return True
    except ValueError:
        return False


def _splice_sites(out_boundary, in_boundary) -> str:
    """IsoForge's per-end exon-boundary flags as one word, as its TSV has it."""
    donor, acceptor = bool(out_boundary), bool(in_boundary)
    return {(True, True): "both", (False, False): "neither",
            (True, False): "donor", (False, True): "acceptor"}[(donor, acceptor)]


class FusionTables:
    """IsoForge's three fusion tables: the calls, their parts, their reads."""

    def __init__(self, source: str):
        self.paths = _table_paths(source)
        self.calls = pd.read_parquet(self.paths["calls"])
        self.parts = pd.read_parquet(self.paths["parts"])
        self.isoforms: Optional[pd.DataFrame] = (
            pd.read_parquet(self.paths["isoforms"]) if "isoforms" in self.paths else None)
        self._reads: Optional[pd.DataFrame] = None

    @property
    def reads(self) -> pd.DataFrame:
        if self._reads is None:
            self._reads = pd.read_parquet(self.paths["reads"])
        return self._reads

    def find(self, key: str) -> Tuple[str, List[str]]:
        """The call ID for *key* (an ID or ``::``-joined genes) and other calls with the same genes.
        """
        if key in set(self.calls["fusion_id"]):
            return key, []
        names = self.calls["gene_names"].map(lambda genes: "::".join(genes))
        matches = self.calls[names == key].sort_values("reads", ascending=False)
        if matches.empty:
            raise ValueError(f"no fusion {key!r} in {os.path.basename(self.paths['calls'])}")
        ids = list(matches["fusion_id"])
        return ids[0], ids[1:]

    def call(self, fusion_id: str) -> FusionCall:
        row = self.calls[self.calls["fusion_id"] == fusion_id].iloc[0]
        parts = self._parts(fusion_id)
        junctions = [
            Junction(a["chromosome"], int(a["junction_out"]), b["chromosome"],
                     int(b["junction_in"]),
                     _splice_sites(a["boundary_out"], b["boundary_in"]))
            for (_, a), (_, b) in zip(parts.iloc[:-1].iterrows(), parts.iloc[1:].iterrows())
        ]
        genes = list(row["gene_names"])
        return FusionCall(
            name="::".join(genes), genes=genes, chromosomes=list(parts["chromosome"]),
            junctions=junctions, reads=int(row["reads"]), status=str(row["filter"]),
            fields={key: row[key] for key in row.index})

    def partners(self, fusion_id: str) -> List[Partner]:
        """Each part as a partner: gene, locus, span, and IsoForge's exon chain."""
        partners = []
        for index, (_, part) in enumerate(self._parts(fusion_id).iterrows()):
            blocks = sorted(zip((int(v) for v in part["block_starts"]),
                                (int(v) for v in part["block_ends"])))
            plus = part["strand"] != "-"
            first = blocks[0][0] if plus else blocks[-1][1]
            last = blocks[-1][1] if plus else blocks[0][0]
            five = int(part["junction_in"]) if pd.notna(part["junction_in"]) else first
            three = int(part["junction_out"]) if pd.notna(part["junction_out"]) else last
            partners.append(Partner(
                index, str(part["gene_name"]), str(part["chromosome"]), str(part["strand"]),
                five, three, gene_start=int(part["gene_start"]),
                gene_end=int(part["gene_end"]), model=blocks))
        return partners

    def fusion_reads(self, fusion_id: str, junctions: int,
                     condition_of: Mapping[str, str]) -> List[FusionRead]:
        """Every supporting read of the call, its pieces grouped by part."""
        rows = self.reads[self.reads["fusion_id"] == fusion_id]
        reads = []
        for (dataset, name), pieces in rows.groupby(["dataset", "read_name"], sort=False):
            by_part: Dict[int, List[Tuple[int, int]]] = {}
            for _, piece in pieces.iterrows():
                if pd.isna(piece["part"]):
                    continue
                by_part.setdefault(int(piece["part"]) - 1, []).extend(
                    zip((int(v) for v in piece["block_starts"]),
                        (int(v) for v in piece["block_ends"])))
            if not by_part:
                continue
            spanned = {j for j in range(junctions) if j in by_part and j + 1 in by_part}
            if bool(pieces["spans_all"].iloc[0]):
                spanned = set(range(junctions))
            first = pieces.iloc[0]
            isoform = first.get("isoform")
            fits = first.get("fits_isoform")
            reads.append(FusionRead(
                name=str(name), condition=condition_of.get(dataset, dataset),
                sample=str(dataset),
                pieces={k: merge_intervals(v) for k, v in by_part.items()},
                junctions=spanned,
                isoform=int(isoform) if isoform is not None and pd.notna(isoform) else None,
                fits=bool(fits) if fits is not None and pd.notna(fits) else None))
        return reads

    def fusion_isoforms(self, fusion_id: str) -> Dict[int, "FusionIsoform"]:
        """IsoForge's transcripts of the call, by number."""
        if self.isoforms is None:
            return {}
        rows = self.isoforms[self.isoforms["fusion_id"] == fusion_id]
        found = {}
        for number, parts in rows.groupby("isoform"):
            parts = parts.sort_values("part")
            ident = parts["isoform_id"].iloc[0] if "isoform_id" in parts else None
            found[int(number)] = FusionIsoform(
                number=int(number), complete=bool(parts["complete"].iloc[0]),
                reads=int(parts["reads"].iloc[0]),
                blocks={int(p) - 1: sorted(zip((int(v) for v in starts), (int(v) for v in ends)))
                        for p, starts, ends in zip(parts["part"], parts["block_starts"],
                                                   parts["block_ends"])},
                isoform_id=str(ident) if ident is not None and pd.notna(ident) else None)
        return found

    def datasets(self, fusion_id: str) -> List[str]:
        rows = self.reads[self.reads["fusion_id"] == fusion_id]
        return list(dict.fromkeys(rows["dataset"]))

    def _parts(self, fusion_id: str) -> pd.DataFrame:
        parts = self.parts[self.parts["fusion_id"] == fusion_id].sort_values("part")
        if parts.empty:
            raise ValueError(f"fusion {fusion_id} has no parts in "
                             f"{os.path.basename(self.paths['parts'])}")
        return parts


# ----------------------------------------------------- one pile per transcript

@dataclass
class FusionIsoform:
    """One transcript of a call as IsoForge built it from the call's reads."""

    number: int
    complete: bool  # covers every partner
    reads: int
    blocks: Dict[int, List[Tuple[int, int]]]  # partner index -> exon blocks
    #: IsoForge's ID for the transcript (``ISOFORF`` and nine digits), when the table has one.
    isoform_id: Optional[str] = None

    @property
    def parts(self) -> Tuple[int, ...]:
        return tuple(sorted(self.blocks))


@dataclass
class ReadGroup:
    """Reads that are one fusion transcript: they join the same partners."""

    name: str
    parts: Tuple[int, ...]
    reads: List[FusionRead]
    complete: bool
    distinct: int = 0
    model: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    #: IsoForge's number for this transcript, when the tables give transcripts.
    isoform: Optional[int] = None
    #: IsoForge's ID for this transcript, when the tables give one.
    isoform_id: Optional[str] = None

    @property
    def junctions(self) -> List[int]:
        """The junctions this group's reads cross."""
        return [p for p in self.parts[:-1] if p + 1 in self.parts]


def _reaches_past(read: FusionRead, partners: Sequence[Partner],
                  junctions: Sequence[Junction], tolerance: int) -> bool:
    """Whether a partial read has sequence the whole fusion does not."""
    first, last = read.parts[0], read.parts[-1]
    if first > 0:  # enters a partner the fusion enters
        partner = partners[first]
        five = _ends(partner, read.pieces[first])[0]
        if partner.is_five_prime_of(five, junctions[first - 1].acceptor - (
                tolerance if partner.strand != "-" else -tolerance)):
            return True
    if last < len(partners) - 1:  # leaves a partner the fusion leaves
        partner = partners[last]
        three = _ends(partner, read.pieces[last])[1]
        if partner.is_five_prime_of(junctions[last].donor + (
                tolerance if partner.strand != "-" else -tolerance), three):
            return True
    return False


def group_reads(partners: Sequence[Partner], junctions: Sequence[Junction],
                reads: Sequence[FusionRead],
                tolerance: int = BREAKPOINT_TOLERANCE,
                isoforms: Optional[Mapping[int, FusionIsoform]] = None) -> List[ReadGroup]:
    """One group per transcript: the whole fusion first, then each partial one."""
    if isoforms:
        by_isoform: Dict[int, List[FusionRead]] = {}
        rest = []
        for read in reads:
            if read.isoform in isoforms:
                by_isoform.setdefault(read.isoform, []).append(read)
            else:
                rest.append(read)
        groups = []
        for number, members in by_isoform.items():
            isoform = isoforms[number]
            distinct = 0 if isoform.complete else sum(
                _reaches_past(r, partners, junctions, tolerance) for r in members)
            groups.append(ReadGroup("::".join(partners[i].gene for i in isoform.parts),
                                    isoform.parts, members, isoform.complete, distinct,
                                    model=dict(isoform.blocks), isoform=number,
                                    isoform_id=isoform.isoform_id))
        groups += group_reads(partners, junctions, rest, tolerance) if rest else []
        groups.sort(key=lambda g: (not g.complete, g.isoform is None, -len(g.reads),
                                   g.isoform or 0))
        return groups

    everything = tuple(range(len(partners)))
    grouped: Dict[Tuple[int, ...], List[FusionRead]] = {}
    for read in reads:
        key = everything if len(read.junctions) == len(junctions) else read.parts
        if len(key) < 2:
            continue
        grouped.setdefault(key, []).append(read)

    groups = []
    for key, members in grouped.items():
        complete = key == everything
        distinct = 0 if complete else sum(
            _reaches_past(r, partners, junctions, tolerance) for r in members)
        groups.append(ReadGroup("::".join(partners[i].gene for i in key), key,
                                members, complete, distinct))
    groups.sort(key=lambda g: (not g.complete, -len(g.reads)))
    return groups
