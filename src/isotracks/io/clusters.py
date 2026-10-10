"""Reader for clustered end-site BED files."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import pandas as pd

SITE_COLUMNS = [
    "chrom", "start", "end", "site", "count", "strand",
    "feature", "gene_id", "motif", "motif_pos",
]

#: Columns before which nothing can be a motif (they are coordinates and ids).
FIRST_MOTIF_COLUMN = 8

_MISSING = {"None", ".", "", "NA", "nan", "NaN"}
_MOTIF = re.compile(r"^[ACGTN]{4,10}$", re.IGNORECASE)


def _parse_motif(fields: Sequence[str], site: int) -> Tuple[Optional[str], Optional[int]]:
    """The sequence motif for a cluster, and where it sits."""
    for value in fields[FIRST_MOTIF_COLUMN:]:
        if value in _MISSING:
            continue
        if "@" in value:
            position, _, motif = value.partition("@")
            if _MOTIF.match(motif):
                try:
                    return motif.upper(), int(position)
                except ValueError:
                    return motif.upper(), site
        elif _MOTIF.match(value):
            return value.upper(), site
    return None, None


class ClusterSiteReader:
    """Clustered end sites, filtered to a region on read.

    Parameters
    ----------
    file_name : str
        A ``tss_clusters.bed``, ``tes_clusters.bed`` or ``polyA_clusters.bed``.
    require_motif : bool
        Drop clusters with no motif.
    """

    def __init__(self, file_name: str, require_motif: bool = False):
        self.file_name = file_name
        self.require_motif = require_motif
        self.region_data: Optional[pd.DataFrame] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> pd.DataFrame:
        self.region_data = self.read_region(region)
        return self.region_data

    def read_region(self, region: Sequence) -> pd.DataFrame:
        """Clusters overlapping ``region[:3]`` as a frame of :data:`SITE_COLUMNS`."""
        chr_name, region_start, region_end = region[0], int(region[1]), int(region[2])
        records: List[dict] = []

        with open(self.file_name) as handle:
            for line in handle:
                if line.startswith(("#", "track", "browser")) or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < FIRST_MOTIF_COLUMN or fields[0] != chr_name:
                    continue
                start, end = int(fields[1]), int(fields[2])
                if end < region_start or start > region_end:
                    continue
                if fields[3] in _MISSING or fields[4] in _MISSING:
                    continue

                site = int(fields[3])
                motif, motif_pos = _parse_motif(fields, site)
                if motif is None and self.require_motif:
                    continue

                records.append({
                    "chrom": fields[0],
                    "start": start,
                    "end": end,
                    "site": site,
                    "count": int(float(fields[4])),
                    "strand": fields[5],
                    "feature": fields[6],
                    "gene_id": fields[7],
                    "motif": motif,
                    "motif_pos": motif_pos,
                })

        return pd.DataFrame(records, columns=SITE_COLUMNS)


class SiteFrameReader:
    """Sites already chosen, handed to each panel as they fall in its window."""

    def __init__(self, sites: pd.DataFrame):
        self.sites = sites if not sites.empty else pd.DataFrame(columns=SITE_COLUMNS)
        self.region_data: Optional[pd.DataFrame] = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def load_region_data(self, region: Sequence, select_features=None) -> pd.DataFrame:
        self.region_data = self.read_region(region)
        return self.region_data

    def read_region(self, region: Sequence) -> pd.DataFrame:
        chr_name, low, high = region[0], *sorted((int(region[1]), int(region[2])))
        sites = self.sites
        inside = (sites["chrom"] == chr_name) & (sites["site"] >= low) & (sites["site"] <= high)
        return sites[inside].reset_index(drop=True)
