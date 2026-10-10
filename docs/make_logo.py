#!/usr/bin/env python3
"""Draw the IsoTracks logo."""

from __future__ import annotations

from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath, text_to_path

OUT = Path(__file__).resolve().parent / "img"

COLORS = {
    "light": dict(ink="#1E293B", model="#0891B2", tail="#EA580C", mod="#7C3AED", depth="#64748B",
                  intron="#94A3B8", rule="#94A3B8", tagline="#94A3B8"),
    "dark": dict(ink="#E6EDF3", model="#22B8D8", tail="#F97316", mod="#A78BFA", depth="#8391A7",
                 intron="#64748B", rule="#8391A7", tagline="#94A3B8"),
}
#: One set for the icon: each clears 3:1 contrast on white and on a dark page (#0D1117).
ICON_COLORS = dict(model="#0E9BBF", tail="#EA580C", mod="#8B5CF6", depth="#6F7F96",
                   intron="#7C8AA0", rule="#7C8AA0")
FONTS = {"name": dict(family="Georgia", weight="bold"),
         "tagline": dict(family="Liberation Sans", weight="normal")}
TAGLINE = "5′ ISOFORMS · READS · MODIFICATIONS · FUSIONS 3′"

W = 340
EXONS = ((24, 60), (104, 132), (164, 200), (232, 276))
#: Modification sites on the model: (x, stem height).
SITES = ((36, 8), (50, 12), (118, 10), (176, 12), (190, 7), (246, 9), (262, 12))
#: Reads, top to bottom: (index of the first exon covered, poly(A) tail length).
READS = ((0, 42), (0, 34), (0, 26), (1, 18), (0, 12), (1, 6))
#: Read depth over each exon.
DEPTH = (9, 13, 13, 16)

ICON = 128
ICON_EXONS = ((8, 40), (62, 96))
ICON_SITES = ((18, 8), (30, 11), (72, 10), (86, 7))
ICON_READS = ((0, 26), (0, 19), (1, 12), (0, 7))


def _prop(kind: str, size: float) -> FontProperties:
    prop = FontProperties(size=size, **FONTS[kind])
    findfont(prop, fallback_to_default=False)
    return prop


def _d(path: MPath, x: float, y: float) -> str:
    """An SVG path from a matplotlib text path, moved to (x, y) with y running down."""
    out = []
    for verts, code in path.iter_segments():
        pts = " ".join(f"{x + verts[i]:.2f} {y - verts[i + 1]:.2f}" for i in range(0, len(verts), 2))
        out.append({MPath.MOVETO: "M", MPath.LINETO: "L", MPath.CURVE3: "Q", MPath.CURVE4: "C"}.get(code, "Z")
                   + ("" if code == MPath.CLOSEPOLY else pts))
    return "".join(out)


def _advance(text: str, prop: FontProperties) -> float:
    return text_to_path.get_text_width_height_descent(text, prop, ismath=False)[0]


def glyphs(text: str, kind: str, size: float, x: float, baseline: float, fill: str,
           tracking: float = 0.0, anchor: str = "start") -> str:
    """*text* as outlines; with *tracking* the letters are set one by one."""
    prop = _prop(kind, size)
    steps = [_advance(c, prop) + tracking for c in text] if tracking else [_advance(text, prop)]
    width = sum(steps) - tracking
    cursor = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
    if not tracking:
        return f'<path d="{_d(TextPath((0, 0), text, prop=prop), cursor, baseline)}" fill="{fill}"/>'
    d = []
    for c, step in zip(text, steps):
        if c.strip():
            d.append(_d(TextPath((0, 0), c, prop=prop), cursor, baseline))
        cursor += step
    return f'<path d="{"".join(d)}" fill="{fill}"/>'


def model(y: float, exons, sites, c: dict, k: float = 1.0) -> list[str]:
    """The isoform: exons joined by intron arcs, with a lollipop at each modification site."""
    h, r = 12 * k, 3 * k
    mid = y + h / 2
    out = []
    for (_, a1), (b0, _) in zip(exons, exons[1:]):
        out.append(f'<path d="M {a1} {mid:.1f} Q {(a1 + b0) / 2} {mid - min(20 * k, (b0 - a1) * 0.62):.1f} '
                   f'{b0} {mid:.1f}" fill="none" stroke="{c["intron"]}" stroke-width="{1.3 * k:.2f}"/>')
    for x0, x1 in exons:
        out.append(f'<rect x="{x0}" y="{y:.1f}" width="{x1 - x0}" height="{h:.1f}" rx="{r:.1f}" '
                   f'fill="{c["model"]}"/>')
    for x, stem in sites:
        top = y - stem * k
        out.append(f'<path d="M {x} {y + 1:.1f} V {top:.1f}" stroke="{c["mod"]}" stroke-width="{1.2 * k:.2f}"/>')
        out.append(f'<circle cx="{x}" cy="{top:.1f}" r="{2.1 * k:.2f}" fill="{c["mod"]}"/>')
    return out


def reads(y: float, exons, rows, c: dict, k: float = 1.0) -> list[str]:
    """The read stack: each read's exon blocks on a hairline, then its poly(A) tail."""
    h, pitch = 3.2 * k, 6.2 * k
    out = []
    end = exons[-1][1]
    for i, (first, tail) in enumerate(rows):
        top = y + i * pitch
        mid = top + h / 2
        out.append(f'<path d="M {exons[first][0]} {mid:.2f} H {end}" stroke="{c["intron"]}" '
                   f'stroke-width="{0.8 * k:.2f}"/>')
        d = "".join(f"M {x0} {top:.2f} h {x1 - x0} v {h:.2f} h {-(x1 - x0)} Z" for x0, x1 in exons[first:])
        out.append(f'<path d="{d}" fill="{c["model"]}"/>')
        out.append(f'<rect x="{end}" y="{top:.2f}" width="{tail}" height="{h:.2f}" '
                   f'rx="{h / 2:.2f}" fill="{c["tail"]}"/>')
        out.append(f'<rect x="{end}" y="{top:.2f}" width="{min(tail / 2, h):.2f}" height="{h:.2f}" '
                   f'fill="{c["tail"]}"/>')
    return out


def coverage(base: float, exons, depth, c: dict) -> list[str]:
    """Read depth over each exon, with an arc for the reads spliced across each intron."""
    out = [f'<rect x="{x0}" y="{base - d}" width="{x1 - x0}" height="{d}" fill="{c["depth"]}"/>'
           for (x0, x1), d in zip(exons, depth)]
    for ((_, a1), da), ((b0, _), db) in zip(zip(exons, depth), zip(exons[1:], depth[1:])):
        apex = base - max(da, db) - 9
        out.append(f'<path d="M {a1} {base - da} Q {(a1 + b0) / 2} {2 * apex - (2 * base - da - db) / 2:.1f} '
                   f'{b0} {base - db}" fill="none" stroke="{c["depth"]}" stroke-width="1.3"/>')
    return out


def axis(y: float, x0: float, x1: float, ticks: int, c: dict, k: float = 1.0) -> str:
    """The coordinate axis: a rule with evenly spaced ticks."""
    step = (x1 - x0) / (ticks - 1)
    marks = "".join(f"M {x0 + i * step:.1f} {y:.1f} v {3.4 * k:.1f} " for i in range(ticks))
    return (f'<path d="M {x0} {y:.1f} H {x1} {marks}" fill="none" stroke="{c["rule"]}" '
            f'stroke-width="{k:.2f}" stroke-linecap="round"/>')


def lockup(mode: str) -> str:
    c = COLORS[mode]
    parts = model(22, EXONS, SITES, c)
    parts += reads(44, EXONS, READS, c)
    parts += coverage(106, EXONS, DEPTH, c)
    parts.append(axis(114, EXONS[0][0], EXONS[-1][1] + READS[0][1], 6, c))
    size, baseline = 44, 171
    x0 = W / 2 - _advance("IsoTracks", _prop("name", size)) / 2
    parts.append(glyphs("Iso", "name", size, x0, baseline, c["ink"]))
    parts.append(glyphs("Tracks", "name", size, x0 + _advance("Iso", _prop("name", size)), baseline, c["tail"]))
    parts.append(glyphs(TAGLINE, "tagline", 7.6, W / 2, baseline + 21, c["tagline"], tracking=1.55,
                        anchor="middle"))
    height = baseline + 32
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'role="img" aria-label="IsoTracks">\n  ' + "\n  ".join(parts) + "\n</svg>\n")


def icon() -> str:
    c, k = ICON_COLORS, 1.6
    parts = model(33, ICON_EXONS, ICON_SITES, c, k)
    parts += reads(65, ICON_EXONS, ICON_READS, c, k)
    parts.append(axis(111, ICON_EXONS[0][0], ICON_EXONS[-1][1] + ICON_READS[0][1], 4, c, k))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON} {ICON}" width="{ICON}" height="{ICON}" '
            f'role="img" aria-label="IsoTracks">\n  ' + "\n  ".join(parts) + "\n</svg>\n")


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two hex colours."""
    def luminance(colour):
        v = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def main() -> None:
    import cairosvg

    OUT.mkdir(parents=True, exist_ok=True)
    for mode, stem in (("light", "isotracks_logo"), ("dark", "isotracks_logo_dark")):
        svg = lockup(mode)
        (OUT / f"{stem}.svg").write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / f"{stem}.png"), output_width=1360)
        print(f"wrote {stem}.svg/.png")
    svg = icon()
    (OUT / "isotracks_icon.svg").write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "isotracks_icon.png"), output_width=512)
    print("wrote isotracks_icon.svg/.png; contrast on white / dark: " + ", ".join(
        f"{k} {contrast(v, '#FFFFFF'):.1f}/{contrast(v, '#0D1117'):.1f}" for k, v in ICON_COLORS.items()))


if __name__ == "__main__":
    main()
