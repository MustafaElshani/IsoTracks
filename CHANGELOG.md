# Changelog

## 1.1.0

- **The opposite strand**: `--antisense` (`antisense=True`) draws the genes transcribed from the opposite strand
  in the window, each in a section below the gene with its isoforms, reads and coverage. `--antisense whole`
  widens the window to draw them whole.
- `--width` takes a width in millimetres (`--width 176`), as its help says. It was read as the name of a
  preset and refused.
- Small exons are drawn at least a hairline wide (0.5 pt times the stroke weight), so at true scale every
  exon of a long transcript shows in its model row as it does in its reads. Widening a narrow intron no
  longer trims the exons beside it away.

## 1.0.0

First public release.

- **Gene figures** from an IsoForge run and its BAMs: poly(A) and transcription start sites, read-end
  density, the gene and its isoform models, read stacks per condition with poly(A) tails, and coverage
  with splice junctions.
- **Fusion figures**: a schematic of the partner genes and the fused transcript, then one panel per
  partner with the fusion's transcripts and reads.
- **RNA modification** and **differential modification** tracks from `modkit pileup` and `modkit dmr pair`.
- **Views**: zoom, a close-up beside the gene, shortened introns and region highlights.
- **Tables**: isoform and exon coordinates, and per-read poly(A) tail lengths.
- **Command line**: `isotracks plot`, `batch`, `info`, `polya` and `index`.
