#!/usr/bin/env python3
"""
Replace uniform light-green map canvas pixels with white (or transparency).

Default reference matches typical Qt/OSM map background (~195,225,195).
Tune --ref and --fuzz if your screenshots differ.

Usage:
  python scripts/replace_map_background.py Academia/images/dataset_conversion_split.png
  python scripts/replace_map_background.py Academia/images/*.png --suffix _whitebg
  python scripts/replace_map_background.py img.png --transparent --out img_nobg.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_rgb(s: str) -> tuple[int, int, int]:
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB must be three integers, e.g. 195,225,195")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description="Replace light green map background with white or alpha.")
    p.add_argument("paths", nargs="+", type=Path, help="Input PNG file(s)")
    p.add_argument(
        "--ref",
        type=parse_rgb,
        default=(195, 225, 195),
        help="Reference background RGB (default: 195,225,195)",
    )
    p.add_argument(
        "--ref2",
        type=parse_rgb,
        default=None,
        help="Optional second reference RGB for slightly different corners (e.g. 156,178,156)",
    )
    p.add_argument(
        "--fuzz",
        type=float,
        default=42.0,
        help="Max Euclidean distance in RGB space to treat as background (default: 42)",
    )
    p.add_argument(
        "--suffix",
        default="_whitebg",
        help="Suffix before .png for output (default: _whitebg). Ignored if --out is set for single file.",
    )
    p.add_argument(
        "--transparent",
        action="store_true",
        help="Write PNG with transparent background instead of white",
    )
    p.add_argument("--out", type=Path, default=None, help="Output path (single input only)")
    args = p.parse_args()

    white = (255, 255, 255)
    refs: list[tuple[int, int, int]] = [args.ref]
    if args.ref2:
        refs.append(args.ref2)

    for path in args.paths:
        if not path.is_file():
            print(f"skip (missing): {path}")
            continue
        im = Image.open(path).convert("RGBA")
        w, h = im.size
        data = im.load()
        fuzz = args.fuzz
        for y in range(h):
            for x in range(w):
                r, g, b, a = data[x, y]
                if a == 0:
                    continue
                rgb = (r, g, b)
                if min(color_distance(rgb, ref) for ref in refs) <= fuzz:
                    if args.transparent:
                        data[x, y] = (255, 255, 255, 0)
                    else:
                        data[x, y] = (*white, a)

        if args.out and len(args.paths) == 1:
            out_path = args.out
        else:
            stem = path.stem
            if stem.endswith(args.suffix):
                stem = stem[: -len(args.suffix)]
            out_path = path.with_name(f"{stem}{args.suffix}.png")

        im.save(out_path, optimize=True)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
