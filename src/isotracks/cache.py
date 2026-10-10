"""A per-gene index for large read-annotation tables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

#: Buckets the rows are split across.
DEFAULT_BUCKETS = 64

#: Rows read from the source TSV at a time while building.
BUILD_CHUNK_ROWS = 500_000

#: Files smaller than this are scanned directly, without a cache.
CACHE_WORTH_IT_BYTES = 256 * 1024 * 1024

CACHE_SUFFIX = ".isotracks-cache"
MANIFEST_NAME = "manifest.json"

#: Environment variable naming a directory for the caches.
CACHE_DIR_ENV = "ISOTRACKS_CACHE_DIR"


def cache_path_for(source: str) -> str:
    """Where the cache for *source* lives."""
    override = os.environ.get(CACHE_DIR_ENV)
    if not override:
        return str(source) + CACHE_SUFFIX
    absolute = os.path.abspath(source)
    digest = hashlib.blake2b(absolute.encode("utf-8"), digest_size=8).hexdigest()
    return os.path.join(override, f"{os.path.basename(absolute)}.{digest}{CACHE_SUFFIX}")


def bucket_of(gene_name: str, buckets: int) -> int:
    """Which bucket a gene's rows belong to."""
    digest = hashlib.blake2b(gene_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


@dataclass(frozen=True)
class CacheManifest:
    """What a cache directory claims about the file it was built from."""

    source: str
    source_size: int
    source_mtime: float
    buckets: int
    rows: int
    columns: List[str]
    gene_column: str
    version: int = 1

    def matches(self, source: str) -> bool:
        try:
            stat = os.stat(source)
        except OSError:
            return False
        return (stat.st_size == self.source_size
                and int(stat.st_mtime) == int(self.source_mtime))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def load(cls, directory: str) -> Optional["CacheManifest"]:
        path = os.path.join(directory, MANIFEST_NAME)
        try:
            with open(path) as handle:
                return cls(**json.load(handle))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None


def find_cache(source: str) -> Optional[CacheManifest]:
    """The usable cache for *source*, or ``None`` if there is none."""
    directory = cache_path_for(source)
    if not os.path.isdir(directory):
        return None
    manifest = CacheManifest.load(directory)
    if manifest is None or not manifest.matches(source):
        return None
    return manifest


def is_worth_caching(source: str) -> bool:
    """Whether *source* is big enough that a cache would pay for itself."""
    try:
        return os.path.getsize(source) >= CACHE_WORTH_IT_BYTES
    except OSError:
        return False


def read_gene(source: str, manifest: CacheManifest, gene_name: str) -> pd.DataFrame:
    """Rows for one gene, read from the bucket that holds it."""
    import pyarrow.parquet as pq

    bucket = os.path.join(
        cache_path_for(source), f"bucket_{bucket_of(gene_name, manifest.buckets):03d}.parquet",
    )
    if not os.path.exists(bucket):
        return pd.DataFrame(columns=manifest.columns)

    table = pq.read_table(bucket, filters=[(manifest.gene_column, "==", gene_name)])
    return table.to_pandas()


def build(
    source: str,
    columns: List[str],
    gene_column: str,
    buckets: int = DEFAULT_BUCKETS,
    chunk_rows: int = BUILD_CHUNK_ROWS,
    progress=None,
) -> CacheManifest:
    """Build the per-gene cache for *source* and return its manifest.

    Parameters
    ----------
    columns : list of str
        Source columns to keep.
    gene_column : str
        The column rows are bucketed and later filtered on.
    progress : callable, optional
        Called with the running row count as each chunk is written.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "building a read-annotation cache needs pyarrow: pip install 'isotracks[parquet]'"
        ) from error

    if gene_column not in columns:
        columns = [*columns, gene_column]

    destination = cache_path_for(source)
    staging = destination + ".building"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    writers: Dict[int, "pq.ParquetWriter"] = {}
    total = 0
    try:
        for chunk in pd.read_csv(
            source, sep="\t", usecols=columns, chunksize=chunk_rows, low_memory=False,
        ):
            chunk = chunk.dropna(subset=[gene_column])
            if chunk.empty:
                continue
            chunk = chunk.astype({gene_column: "string"})
            assignments = chunk[gene_column].map(lambda name: bucket_of(str(name), buckets))
            for index, rows in chunk.groupby(assignments, sort=False):
                table = pa.Table.from_pandas(rows, preserve_index=False)
                writer = writers.get(int(index))
                if writer is None:
                    writer = pq.ParquetWriter(
                        os.path.join(staging, f"bucket_{int(index):03d}.parquet"),
                        table.schema, compression="zstd",
                    )
                    writers[int(index)] = writer
                writer.write_table(table)
            total += len(chunk)
            if progress is not None:
                progress(total)
    finally:
        for writer in writers.values():
            writer.close()

    stat = os.stat(source)
    manifest = CacheManifest(
        source=os.path.abspath(source),
        source_size=stat.st_size,
        source_mtime=stat.st_mtime,
        buckets=buckets,
        rows=total,
        columns=list(columns),
        gene_column=gene_column,
    )
    with open(os.path.join(staging, MANIFEST_NAME), "w") as handle:
        handle.write(manifest.to_json())

    shutil.rmtree(destination, ignore_errors=True)
    os.replace(staging, destination)
    return manifest


def remove(source: str) -> bool:
    """Delete the cache for *source*."""
    directory = cache_path_for(source)
    if not os.path.isdir(directory):
        return False
    shutil.rmtree(directory)
    return True
