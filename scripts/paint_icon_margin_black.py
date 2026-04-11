#!/usr/bin/env python3
"""
Paint edge-connected near-white margins black in assets/icon.png (rounded-square
icons often ship with a white square canvas). In-place overwrite by default.

Uses PySide6 (same as the GUI).

  .venv/bin/python3 scripts/paint_icon_margin_black.py
  .venv/bin/python3 scripts/paint_icon_margin_black.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ICON = _ROOT / "assets" / "icon.png"


def paint_margin_black(
    path: Path,
    *,
    tol: int = 36,
    dry_run: bool = False,
) -> int:
    try:
        from PySide6.QtGui import QColor, QImage
    except ModuleNotFoundError:
        print(
            "PySide6 required, e.g. .venv/bin/python3 scripts/paint_icon_margin_black.py",
            file=sys.stderr,
        )
        return 1

    img = QImage(str(path))
    if img.isNull():
        print(f"Failed to load: {path}", file=sys.stderr)
        return 1
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    ref = QColor(img.pixel(0, 0))
    r0, g0, b0 = ref.red(), ref.green(), ref.blue()

    def similar(x: int, y: int) -> bool:
        c = QColor(img.pixel(x, y))
        return (
            abs(c.red() - r0) + abs(c.green() - g0) + abs(c.blue() - b0) <= tol
        )

    seen = bytearray(w * h)

    def ixy(x: int, y: int) -> int:
        return y * w + x

    q: deque[tuple[int, int]] = deque()

    def try_push(x: int, y: int) -> None:
        if not (0 <= x < w and 0 <= y < h):
            return
        k = ixy(x, y)
        if seen[k]:
            return
        if not similar(x, y):
            return
        seen[k] = 1
        q.append((x, y))

    for x in range(w):
        try_push(x, 0)
        try_push(x, h - 1)
    for y in range(h):
        try_push(0, y)
        try_push(w - 1, y)

    cells: list[tuple[int, int]] = []
    while q:
        x, y = q.popleft()
        cells.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            try_push(nx, ny)

    painted = len(cells)
    print(f"{path.name}: would paint {painted} px to black (tol={tol}, ref=({r0},{g0},{b0}))")
    if dry_run:
        print("Dry run — file not modified.")
        return 0

    black = QColor(0, 0, 0)
    for x, y in cells:
        img.setPixelColor(x, y, black)
    if not img.save(str(path), "PNG"):
        print(f"Failed to save: {path}", file=sys.stderr)
        return 1
    print(f"Saved {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=_DEFAULT_ICON,
        help=f"PNG path (default: {_DEFAULT_ICON})",
    )
    ap.add_argument(
        "--tol",
        type=int,
        default=36,
        help="Manhattan distance from top-left for 'background' (default: 36)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    if not ns.path.is_file():
        print(f"Missing file: {ns.path}", file=sys.stderr)
        return 1
    return paint_margin_black(ns.path, tol=ns.tol, dry_run=ns.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
