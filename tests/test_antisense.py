"""Genes on the opposite strand, drawn below the gene's own tracks."""

from __future__ import annotations

import pytest

from isotracks import IsoTracks
from isotracks.cli import main
from isotracks.tracks import CoverageTrack, GeneModelTrack, ReadTrack, TranscriptTrack

from .conftest import GENE


def _rows(view, kind):
    return [t for t in view.canvas.tracks if type(t) is kind]


def _drawn(antisense_kwargs, mode, **extra):
    view = IsoTracks(**{**antisense_kwargs, "antisense": mode, **extra})
    view.plot()
    return view


def test_nothing_from_the_opposite_strand_by_default(antisense_kwargs):
    view = _drawn(antisense_kwargs, None)
    assert [t.label for t in _rows(view, GeneModelTrack)] == [GENE]
    assert [t.label for t in _rows(view, TranscriptTrack)] == ["TX_LONG", "TX_SHORT"]


def test_the_opposite_strand_gene_in_the_window_gets_a_section_below(antisense_kwargs):
    plain = _drawn(antisense_kwargs, None)
    view = _drawn(antisense_kwargs, "window")
    assert view.canvas.region == plain.canvas.region
    genes = _rows(view, GeneModelTrack)
    assert [t.label for t in genes] == [GENE, "ANTI (-)"]
    assert genes[1].config["rule_above"]
    assert [t.label for t in _rows(view, TranscriptTrack)] == ["TX_LONG", "TX_SHORT", "AS_LONG"]
    kinds = [type(t).__name__ for t in view.canvas.tracks]
    section = [i for i, kind in enumerate(kinds) if kind == "GeneModelTrack"][1]
    assert kinds[section - 1] == "CoverageTrack"
    assert kinds[section:].count("CoverageTrack") == 2


def test_whole_widens_the_window_to_draw_the_opposite_gene_whole(antisense_kwargs):
    view = _drawn(antisense_kwargs, "whole")
    _, start, end, strand = view.canvas.region
    assert strand == "+" and start <= 1_000 and end >= 9_400
    assert [t.label for t in _rows(view, TranscriptTrack)][2:] == ["AS_LONG", "AS_SHORT"]


def test_opposite_strand_reads_are_their_own_and_reversed(antisense_kwargs):
    view = _drawn(antisense_kwargs, "window")
    stacks = _rows(view, ReadTrack)
    own, opposite = stacks[:4], stacks[4:]
    assert opposite, "the opposite gene's reads are drawn"
    for track in own:
        assert all("|TX_" in name for name in track.region_data.df["read_name"])
    for track in opposite:
        assert all("|AS_LONG" in name for name in track.region_data.df["read_name"])
        for feature in track.region_data:
            tails = [c for c in feature.children if c.type == "polyA"]
            exons = [c for c in feature.children if c.type != "polyA"]
            assert tails and tails[0].end < min(e.start for e in exons), "tail at the 3' end, left"


def test_opposite_isoforms_take_colours_of_their_own(antisense_kwargs):
    view = _drawn(antisense_kwargs, "whole")
    colours = [t.config["style"]["exon"]["facecolor"] for t in _rows(view, TranscriptTrack)]
    assert len(set(colours)) == len(colours)


def test_each_gene_has_its_own_depth_scale(antisense_kwargs):
    view = _drawn(antisense_kwargs, "window")
    tracks = _rows(view, CoverageTrack)
    own, opposite = tracks[:2], tracks[2:]
    assert own[0].ylim == own[1].ylim and opposite[0].ylim == opposite[1].ylim
    assert own[0].ylim != opposite[0].ylim


def test_shortened_introns_keep_the_opposite_genes_exons(antisense_kwargs):
    view = _drawn(antisense_kwargs, "whole", verbose=False)
    view.plot(introns="fixed")
    intron_map = view.canvas.intron_map
    assert intron_map.breaks, "TESTG's introns are shortened"
    for start, end in [(4_500, 4_700), (7_400, 7_700), (8_000, 8_300), (9_000, 9_400)]:
        assert not any(a <= end and b >= start for a, b in intron_map.breaks)


def test_antisense_is_window_whole_or_true(antisense_kwargs):
    assert IsoTracks(**{**antisense_kwargs, "antisense": True}).antisense == "window"
    assert IsoTracks(**{**antisense_kwargs, "antisense": False}).antisense is None
    with pytest.raises(ValueError, match="antisense"):
        IsoTracks(**{**antisense_kwargs, "antisense": "both"})


@pytest.mark.parametrize("flag, sections", [([], 1), (["--antisense"], 2),
                                            (["--antisense", "whole"], 2)])
def test_the_cli_draws_the_opposite_strand_on_request(antisense_kwargs, tmp_path,
                                                      monkeypatch, flag, sections):
    drawn = {}
    real = IsoTracks.plot

    def spy(self, *args, **kwargs):
        drawn["view"] = self
        return real(self, *args, **kwargs)

    monkeypatch.setattr(IsoTracks, "plot", spy)
    bam = ";".join(f"{c}:{','.join(p)}" for c, p in antisense_kwargs["bam_files"].items())
    assert main(["plot", "--gene", GENE, "--bam", bam,
                 "--annotation-gtf", antisense_kwargs["annotation_gtf"],
                 "--read-annot", antisense_kwargs["read_annot"], "--min-reads", "1",
                 *flag, "--output", str(tmp_path / "out.png"), "--quiet"]) == 0
    assert len(_rows(drawn["view"], GeneModelTrack)) == sections
