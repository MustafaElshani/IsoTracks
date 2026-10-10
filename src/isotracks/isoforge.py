"""Locate the output files of an IsoForge run from its prefix."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Sequence, Union

#: Argument name -> file suffixes after the run prefix, in order of preference.
OUTPUT_SUFFIXES = {
    "annotation_gtf": ("_isoforge.gtf",),
    "read_annot": ("_isoforge_read_annot.parquet", "_isoforge_read_annot.tsv"),
    "bed_file": ("_isoforge_sqanti3.bed",),
    "polya_clusters": ("_TSS_TES_Clusters/tes_clusters.bed",),
    "tss_clusters": ("_TSS_TES_Clusters/tss_clusters.bed",),
    "fusion_table": ("_isoforge_fusions.parquet",),
}
#: Directory of the alignments ``isoforge annotate --write_bam`` keeps, after the prefix.
ALIGNMENTS_SUFFIX = "_alignments"

_ALL_SUFFIXES = tuple(s for suffixes in OUTPUT_SUFFIXES.values() for s in suffixes)


def run_prefix(path: str) -> str:
    """The prefix of an IsoForge run, given the prefix, one of its files or its directory."""
    path = os.fspath(path)
    if os.path.isdir(path):
        prefixes = {
            name[: -len(suffix)]
            for suffix in _ALL_SUFFIXES if "/" not in suffix
            for name in glob.glob(os.path.join(path, "*" + suffix))
        }
        if len(prefixes) != 1:
            found = ", ".join(sorted(os.path.basename(p) for p in prefixes)) or "none"
            raise ValueError(
                f"{path!r} must hold one IsoForge run (found: {found}); pass its prefix instead")
        return prefixes.pop()
    for suffix in _ALL_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def condition_of(sample: str) -> str:
    """The condition of a sample: its name up to the first underscore, as IsoForge reads it."""
    return sample.split("_", 1)[0]


def group_by_condition(bam_files: Sequence[str]) -> Dict[str, List[str]]:
    """BAM files grouped by the condition in their file names."""
    groups: Dict[str, List[str]] = {}
    for path in sorted(bam_files):
        sample = os.path.basename(path).rsplit(".", 1)[0]
        groups.setdefault(condition_of(sample), []).append(path)
    return groups


def find_outputs(path: str) -> Dict[str, Union[str, Dict[str, List[str]]]]:
    """The files of the IsoForge run at *path*, keyed by the argument each one fills."""
    prefix = run_prefix(path)
    found: Dict[str, Union[str, Dict[str, List[str]]]] = {}
    for key, suffixes in OUTPUT_SUFFIXES.items():
        for suffix in suffixes:
            if os.path.exists(prefix + suffix):
                found[key] = prefix if key == "fusion_table" else prefix + suffix
                break
    alignments = glob.glob(os.path.join(prefix + ALIGNMENTS_SUFFIX, "*.bam"))
    if alignments:
        found["bam_files"] = group_by_condition(alignments)
    if not found:
        raise ValueError(
            f"no IsoForge output at {path!r}: expected files such as "
            f"{os.path.basename(prefix)}_isoforge.gtf beside it")
    return found
