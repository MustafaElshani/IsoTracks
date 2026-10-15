"""Annotation inputs: GTF gene models, read-to-transcript tables, BED12 filtering."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from . import cache as cache_module

#: Keep only the whole reads of their transcript (``cluster_supported``).
SUPPORTED_ONLY = "supported"
#: Keep every read in the table.
ALL_READS = "all"
READ_FILTERS = (SUPPORTED_ONLY, ALL_READS)

#: The column IsoForge marks whole reads of a transcript in.
SUPPORT_COLUMN = "cluster_supported"

#: Read-annotation column spellings we accept, mapped to the canonical name.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "read_name": ("read_name", "read_id", "readname"),
    # ``transcript_ID``, the numeric database id, stays last.
    "transcript_id": ("transcript_id", "annot_transcript_id", "transcript_ID"),
    "transcript_name": ("annot_transcript_name", "transcript_name"),
    "gene_name": ("annot_gene_name", "gene_name"),
    "sample": ("sample", "dataset"),
    "chromosome": ("chromosome", "Chromosome", "chrom", "chr"),
    "strand": ("strand", "Strand"),
    "polya_length": ("polya_length", "polyA_length"),
    SUPPORT_COLUMN: ("cluster_supported",),
    "fraction_As": ("fraction_As", "fraction_as"),
    "polya_signal": ("polya_signal", "polyA_signal"),
}

_GTF_CACHE: Dict[Tuple[str, float, str], "GeneRecords"] = {}
_GTF_CACHE_LIMIT = 32
#: Per GTF file: chromosome -> sorted ``(start, end, strand, gene_name)`` of its genes.
_SPAN_CACHE: Dict[Tuple[str, float], Dict[str, List[Tuple[int, int, str, str]]]] = {}


def parse_gtf_attributes(text: str) -> Dict[str, str]:
    """GTF column 9 as a dict, quotes stripped."""
    attributes: Dict[str, str] = {}
    for item in text.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item or " " not in item:
            continue
        key, _, value = item.partition(" ")
        attributes[key] = value.strip().strip('"')
    return attributes


@dataclass(frozen=True)
class Locus:
    """A placed gene."""

    gene_name: str
    chromosome: str
    start: int
    end: int
    strand: str
    source: str = ""

    @property
    def region(self) -> Tuple[str, int, int, str]:
        return (self.chromosome, self.start, self.end, self.strand)


@dataclass
class GeneRecords:
    """Everything one GTF says about one gene, gathered in a single pass."""

    gene_name: str
    gene: Optional[Locus] = None
    transcripts: pd.DataFrame = field(default_factory=pd.DataFrame)
    exons: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.gene is None and self.transcripts.empty and not self.exons

    def transcript_extent(self) -> Optional[Locus]:
        """Span of the transcripts actually present, rather than the gene feature."""
        if self.transcripts.empty:
            return None
        row = self.transcripts.iloc[0]
        return Locus(
            self.gene_name, str(row["chromosome"]),
            int(self.transcripts["start"].min()), int(self.transcripts["end"].max()),
            str(row["strand"]), source="transcripts",
        )

    def representative(self) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        """The gene's representative isoform: MANE Select, else Ensembl canonical, else longest."""
        if not self.exons:
            return None, []
        flags = {}
        if not self.transcripts.empty and "mane" in self.transcripts:
            flags = {str(row.transcript_id): (bool(row.mane), bool(row.canonical))
                     for row in self.transcripts.itertuples()}

        def rank(tid):
            mane, canonical = flags.get(tid, (False, False))
            return mane, canonical, sum(end - start + 1 for start, end in self.exons[tid])

        tid = max(sorted(self.exons), key=rank)
        return tid, sorted(self.exons[tid])

    def exons_in_transcription_order(self, transcript_id: Optional[str] = None
                                     ) -> Dict[str, List[Tuple[int, int]]]:
        """Exons numbered 5' to 3', so exon 1 is the TSS on either strand."""
        strand = self.gene.strand if self.gene else "+"
        out: Dict[str, List[Tuple[int, int]]] = {}
        for tid, spans in self.exons.items():
            if transcript_id and tid != transcript_id:
                continue
            ordered = sorted(spans, key=lambda span: span[0])
            if strand == "-":
                ordered.reverse()
            out[tid] = ordered
        return out


class GtfIndex:
    """Reads one gene's worth of records out of a GTF in a single pass."""

    def __init__(self, path: str):
        self.path = path

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GtfIndex({self.path!r})"

    def gene_records(self, gene_name: str) -> GeneRecords:
        if not self.path or not os.path.exists(self.path):
            return GeneRecords(gene_name)

        key = (os.path.abspath(self.path), os.path.getmtime(self.path), gene_name)
        cached = _GTF_CACHE.get(key)
        if cached is not None:
            return cached

        records = self._scan(gene_name)
        if len(_GTF_CACHE) >= _GTF_CACHE_LIMIT:
            _GTF_CACHE.pop(next(iter(_GTF_CACHE)))
        _GTF_CACHE[key] = records
        return records

    def genes_overlapping(self, chromosome: str, start: int, end: int,
                          strand: Optional[str] = None) -> List[str]:
        """Names of the genes whose transcripts overlap ``start..end``, optionally on one strand."""
        return [name for low, high, gene_strand, name in self._gene_spans().get(chromosome, [])
                if low <= end and high >= start and (strand is None or gene_strand == strand)]

    def _gene_spans(self) -> Dict[str, List[Tuple[int, int, str, str]]]:
        """Every gene's transcribed span, read once per file."""
        if not self.path or not os.path.exists(self.path):
            return {}
        key = (os.path.abspath(self.path), os.path.getmtime(self.path))
        if key not in _SPAN_CACHE:
            spans: Dict[Tuple[str, str, str], List[int]] = {}
            with open(self.path) as handle:
                for line in handle:
                    if line[0] == "#":
                        continue
                    fields = line.split("\t", 8)
                    if len(fields) < 9 or fields[2] != "transcript":
                        continue
                    name = parse_gtf_attributes(fields[8]).get("gene_name")
                    if not name:
                        continue
                    low, high = int(fields[3]), int(fields[4])
                    span = spans.setdefault((fields[0], fields[6], name), [low, high])
                    span[0], span[1] = min(span[0], low), max(span[1], high)
            by_chromosome: Dict[str, List[Tuple[int, int, str, str]]] = {}
            for (chromosome, strand, name), (low, high) in spans.items():
                by_chromosome.setdefault(chromosome, []).append((low, high, strand, name))
            _SPAN_CACHE[key] = {c: sorted(v) for c, v in by_chromosome.items()}
        return _SPAN_CACHE[key]

    def _scan(self, gene_name: str) -> GeneRecords:
        needle = f'"{gene_name}"'
        gene: Optional[Locus] = None
        transcript_rows: List[dict] = []
        exons: Dict[str, List[Tuple[int, int]]] = {}

        with open(self.path) as handle:
            for line in handle:
                if line[0] == "#" or needle not in line:
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    continue
                feature = fields[2]
                if feature not in ("gene", "transcript", "exon"):
                    continue
                attributes = parse_gtf_attributes(fields[8])
                if attributes.get("gene_name") != gene_name:
                    continue

                chromosome, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
                if feature == "gene":
                    if gene is None:
                        gene = Locus(gene_name, chromosome, start, end, strand, source="gene")
                elif feature == "transcript":
                    transcript_rows.append({
                        "transcript_id": attributes.get("transcript_id", ""),
                        "transcript_name": attributes.get("transcript_name", ""),
                        "chromosome": chromosome,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "tss_site": _as_int(
                            attributes.get("tss_site"),
                            default=start if strand == "+" else end),
                        "polya_site": _as_int(
                            attributes.get("polyA_site") or attributes.get("polya_site"),
                            default=end if strand == "+" else start),
                        "mane": '"MANE_Select"' in fields[8],
                        "canonical": '"Ensembl_canonical"' in fields[8],
                    })
                else:
                    tid = attributes.get("transcript_id")
                    if tid:
                        exons.setdefault(tid, []).append((start, end))

        records = GeneRecords(
            gene_name=gene_name,
            gene=gene,
            transcripts=pd.DataFrame(transcript_rows) if transcript_rows else pd.DataFrame(
                columns=["transcript_id", "transcript_name", "chromosome",
                         "start", "end", "strand", "tss_site", "polya_site",
                         "mane", "canonical"]),
            exons=exons,
        )
        if records.gene is None:
            records.gene = records.transcript_extent()
        return records

    def transcript_name_map(self, gene_name: str) -> Dict[str, str]:
        """``transcript_id`` to ``transcript_name`` for one gene."""
        records = self.gene_records(gene_name)
        if records.transcripts.empty:
            return {}
        pairs = records.transcripts[["transcript_id", "transcript_name"]]
        return {
            str(tid): str(name)
            for tid, name in pairs.itertuples(index=False)
            if tid and name
        }


def _as_int(value, default=None) -> Optional[int]:
    if value in (None, "", "None", "."):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a read-annotation frame's columns to the canonical spellings."""
    present = set(df.columns)
    renames: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in present:
            continue
        for alias in aliases:
            if alias in present:
                renames[alias] = canonical
                break
    return df.rename(columns=renames) if renames else df


def useful_columns(available: Iterable[str]) -> List[str]:
    """The subset of *available* worth reading, in file order."""
    wanted = {alias for aliases in COLUMN_ALIASES.values() for alias in aliases}
    return [name for name in available if name in wanted]


@dataclass
class GeneReads:
    """The reads one annotation table assigns to one gene."""

    gene_name: str
    frame: pd.DataFrame
    reads_by_transcript: Dict[str, Set[str]]

    @property
    def read_names(self) -> Set[str]:
        return set(self.frame["read_name"]) if not self.frame.empty else set()

    def counts(self) -> Dict[str, int]:
        return {tid: len(reads) for tid, reads in self.reads_by_transcript.items()}

    def restrict_to(self, transcripts: Set[str]) -> "GeneReads":
        """Keep only reads assigned to one of *transcripts*."""
        if self.frame.empty or not transcripts:
            return self
        frame = self.frame[self.frame["transcript_id"].isin(transcripts)]
        kept = {
            tid: reads for tid, reads in self.reads_by_transcript.items()
            if tid in transcripts
        }
        return GeneReads(self.gene_name, frame, kept)

    def polya_lengths(self) -> Dict[str, int]:
        """Per-read tail lengths, when the table carries them (IsoForge does)."""
        if "polya_length" not in self.frame.columns:
            return {}
        lengths = self.frame.dropna(subset=["polya_length"])
        return {
            str(name): int(value)
            for name, value in zip(lengths["read_name"], lengths["polya_length"])
        }

    def read_index(self) -> Dict[str, str]:
        """``read id -> transcript_id``, keyed the way BAM names compare."""
        from .io.longreads import read_key

        return {
            read_key(read): tid
            for tid, reads in self.reads_by_transcript.items()
            for read in reads
        }

    def counts_by_condition(self, sample_to_condition: Dict[str, str]) -> pd.DataFrame:
        """Reads per transcript per condition, summed over replicates."""
        if self.frame.empty or "sample" not in self.frame.columns:
            return pd.DataFrame(columns=["transcript_id", "condition", "count"])
        frame = self.frame.copy()
        frame["condition"] = frame["sample"].map(sample_to_condition)
        frame = frame.dropna(subset=["condition"])
        if frame.empty:
            return pd.DataFrame(columns=["transcript_id", "condition", "count"])
        return (
            frame.groupby(["transcript_id", "condition"], observed=True)
            .size().reset_index(name="count")
        )

    @property
    def samples(self) -> List[str]:
        if self.frame.empty or "sample" not in self.frame.columns:
            return []
        return sorted(str(s) for s in self.frame["sample"].dropna().unique())


class ReadAnnotation:
    """A read-to-transcript table, read one gene at a time."""

    CHUNK_ROWS = 500_000

    def __init__(self, path: str, use_cache: bool = True, verbose: bool = True,
                 read_filter: str = SUPPORTED_ONLY,
                 max_frac_a: Optional[float] = None):
        self.path = path
        self.use_cache = use_cache
        self.verbose = verbose
        self.read_filter = read_filter
        self.max_frac_a = max_frac_a
        self._cache: Dict[Tuple[str, Optional[str], Optional[str]], GeneReads] = {}
        self._warned_about_cache = False

    @property
    def is_parquet(self) -> bool:
        return self.path.lower().endswith((".parquet", ".pq"))

    @property
    def cache_manifest(self):
        """The usable on-disk cache for this table, if there is one."""
        if not self.use_cache:
            return None
        return cache_module.find_cache(self.path)

    def build_cache(self, buckets: int = cache_module.DEFAULT_BUCKETS, progress=None):
        """Build the per-gene cache for this table (see :mod:`isotracks.cache`)."""
        header = pd.read_csv(self.path, sep="\t", nrows=0)
        gene_column = next(
            (name for name in COLUMN_ALIASES["gene_name"] if name in header.columns), None,
        )
        if gene_column is None:
            raise ValueError(
                f"{self.path} has no gene-name column; expected one of "
                f"{COLUMN_ALIASES['gene_name']}"
            )
        return cache_module.build(
            self.path, useful_columns(header.columns), gene_column,
            buckets=buckets, progress=progress,
        )

    def _note_missing_cache(self) -> None:
        """Say once that a cache would turn this scan into a seek."""
        if self._warned_about_cache or not self.verbose:
            return
        self._warned_about_cache = True
        if not self.use_cache or not cache_module.is_worth_caching(self.path):
            return
        size_gb = os.path.getsize(self.path) / 1e9
        print(
            f"  scanning {size_gb:.1f} GB of read annotation; index it once with\n"
            f"    isotracks index --read-annot {self.path}"
        )

    def for_gene(self, gene_name: str, chromosome: Optional[str] = None,
                 strand: Optional[str] = None) -> GeneReads:
        """Rows for one gene, optionally confined to one chromosome and strand."""
        key = (gene_name, chromosome, strand)
        if key in self._cache:
            return self._cache[key]

        frame = self._read_gene(gene_name, chromosome, strand)
        frame = self._drop_unsupported(frame)
        reads_by_transcript: Dict[str, Set[str]] = {}
        if not frame.empty:
            for tid, group in frame.groupby("transcript_id", observed=True):
                reads_by_transcript[str(tid)] = set(group["read_name"])

        result = GeneReads(gene_name, frame, reads_by_transcript)
        self._cache[key] = result
        return result

    def _drop_unsupported(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the read filter, saying once how much it removed."""
        if frame.empty:
            return frame
        kept = apply_read_filter(frame, self.read_filter, self.max_frac_a)
        dropped = len(frame) - len(kept)
        if dropped and self.verbose:
            print(
                f"  dropped {dropped:,} of {len(frame):,} reads that are not whole reads "
                "of their transcript (read_filter='all' keeps them)"
            )
        return kept

    # ----------------------------------------------------------- file access

    def _read_gene(self, gene_name: str, chromosome: Optional[str],
                   strand: Optional[str]) -> pd.DataFrame:
        if not self.path or not os.path.exists(self.path):
            return _empty_reads()

        manifest = self.cache_manifest
        if manifest is not None:
            frame = canonical_columns(cache_module.read_gene(self.path, manifest, gene_name))
        elif self.is_parquet:
            frame = self._read_gene_parquet(gene_name)
        else:
            self._note_missing_cache()
            frame = self._read_gene_tsv(gene_name)
        return _tidy_gene_frame(frame, chromosome, strand)

    def _read_gene_parquet(self, gene_name: str) -> pd.DataFrame:
        import pyarrow.parquet as pq

        schema_names = pq.ParquetFile(self.path).schema_arrow.names
        columns = useful_columns(schema_names)
        gene_column = next(
            (name for name in COLUMN_ALIASES["gene_name"] if name in schema_names), None,
        )
        filters = [(gene_column, "==", gene_name)] if gene_column else None
        table = pq.read_table(self.path, columns=columns or None, filters=filters)
        frame = canonical_columns(table.to_pandas())
        if gene_column is None and "gene_name" in frame.columns:
            frame = frame[frame["gene_name"] == gene_name]
        return frame

    def _read_gene_tsv(self, gene_name: str) -> pd.DataFrame:
        header = pd.read_csv(self.path, sep="\t", nrows=0)
        columns = useful_columns(header.columns)
        gene_column = next(
            (name for name in COLUMN_ALIASES["gene_name"] if name in header.columns), None,
        )
        if gene_column is None:
            raise ValueError(
                f"{self.path} has no gene-name column; expected one of "
                f"{COLUMN_ALIASES['gene_name']}"
            )

        matches: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            self.path, sep="\t", usecols=columns or None,
            chunksize=self.CHUNK_ROWS, low_memory=False,
        ):
            hit = chunk[chunk[gene_column] == gene_name]
            if not hit.empty:
                matches.append(canonical_columns(hit))
        return pd.concat(matches, ignore_index=True) if matches else _empty_reads()


def apply_read_filter(
    frame: pd.DataFrame,
    how: str,
    max_frac_a: Optional[float] = None,
) -> pd.DataFrame:
    """Keep the whole reads of their transcript."""
    if how not in READ_FILTERS:
        raise ValueError(f"unknown read filter {how!r}; use one of {READ_FILTERS}")
    if how == ALL_READS or frame.empty:
        return frame

    if SUPPORT_COLUMN in frame.columns:
        frame = frame[frame[SUPPORT_COLUMN].astype("boolean").fillna(False)]
    if max_frac_a is not None and {"fraction_As", "polya_signal"} <= set(frame.columns):
        primed = (frame["fraction_As"] > max_frac_a) & frame["polya_signal"].isna()
        frame = frame[~primed]
    return frame


def _empty_reads() -> pd.DataFrame:
    return pd.DataFrame(columns=["read_name", "transcript_id", "sample"])


def _tidy_gene_frame(frame: pd.DataFrame, chromosome: Optional[str],
                     strand: Optional[str]) -> pd.DataFrame:
    if frame.empty:
        return _empty_reads()
    if chromosome and "chromosome" in frame.columns:
        frame = frame[frame["chromosome"].astype(str) == str(chromosome)]
    if strand and "strand" in frame.columns:
        frame = frame[frame["strand"].astype(str) == str(strand)]
    if frame.empty:
        return _empty_reads()

    missing = [c for c in ("read_name", "transcript_id") if c not in frame.columns]
    if missing:
        raise ValueError(f"read annotation is missing required column(s): {', '.join(missing)}")

    frame = frame.dropna(subset=["read_name", "transcript_id"]).copy()
    frame["read_name"] = frame["read_name"].astype(str)
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if "sample" in frame.columns:
        frame["sample"] = frame["sample"].astype(str)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------- BED12

BED12_COLUMNS = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand",
    "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts",
]


def load_bed(path: str) -> pd.DataFrame:
    """Read a BED12 file, tolerating extra trailing columns."""
    frame = pd.read_csv(path, sep="\t", header=None, comment="#")
    width = min(len(frame.columns), len(BED12_COLUMNS))
    frame = frame.iloc[:, :width]
    frame.columns = BED12_COLUMNS[:width]
    return frame


def select_bed_transcripts(
    frame: pd.DataFrame,
    transcript_ids: Sequence[str],
    name_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Rows for *transcript_ids*, keying the name column back to transcript ids."""
    wanted = set(transcript_ids)
    matched = frame["name"].isin(wanted)
    if matched.all() or not name_map:
        return frame[matched].copy()

    names_wanted = {name_map[tid] for tid in wanted if tid in name_map}
    if not names_wanted:
        return frame[matched].copy()

    selected = frame[matched | frame["name"].isin(names_wanted)].copy()
    name_to_id = {name: tid for tid, name in name_map.items() if tid in wanted}
    selected["name"] = selected["name"].map(lambda value: name_to_id.get(value, value))
    return selected


def map_samples_to_conditions(
    bam_files: Dict[str, Sequence[str]],
    known_samples: Sequence[str],
) -> Dict[str, str]:
    """Match each BAM file to the sample name the annotation uses."""
    ordered = sorted((str(s) for s in known_samples), key=len, reverse=True)
    mapping: Dict[str, str] = {}
    for condition, paths in bam_files.items():
        for path in paths:
            stem = os.path.basename(path)
            for suffix in (".bam", ".cram", ".sam"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            match = next((s for s in ordered if s and (stem.startswith(s) or s in stem)), None)
            if match is not None:
                mapping[match] = condition
    return mapping
