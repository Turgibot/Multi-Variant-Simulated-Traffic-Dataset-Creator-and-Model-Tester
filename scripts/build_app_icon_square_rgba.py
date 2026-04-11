#!/usr/bin/env python3
"""
Regenerate assets/app_icon_square_rgba.png from assets/traffic_app_icon.png.

The GUI and Linux installer prefer assets/icon.png when that file exists; this
script is only needed for the older generated fallback asset.

Strips pale side columns, classifies pixels as UI/background (sky, blue card,
pale chart fill) vs. content (road, vehicles, graph), paints background to black,
tight-bounds the content, centers on a square black canvas.

Run from repo root (PySide6 required, e.g. .venv):

  .venv/bin/python3 scripts/build_app_icon_square_rgba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "assets" / "traffic_app_icon.png"
_DST = _ROOT / "assets" / "app_icon_square_rgba.png"

_SKY_REF = (183, 196, 228)
_PANEL_REF = (33, 50, 130)


def _manhattan(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _is_background_rgb(r: int, g: int, b: int) -> bool:
    """True for sky margin, dark blue plate, and pale blue chart fill — not road/graph/vehicles."""
    # Slightly cool near-whites (card chrome), not neutral (253,253,253) road
    if r >= 244 and g >= 244 and b >= 248 and (b > r + 3 or b > g + 3):
        return True
    # Pale blue interior fills (stronger B than R)
    if (
        195 <= r < 248
        and 200 <= g < 252
        and 228 <= b <= 255
        and not (r >= 248 and g >= 248)
        and b > r + 15
    ):
        return True
    if _manhattan((r, g, b), _SKY_REF) <= 75:
        return True
    if b > 95 and _manhattan((r, g, b), _PANEL_REF) <= 85:
        return True
    if b > 100 and _manhattan((r, g, b), (45, 70, 155)) <= 55:
        return True
    if b > 100 and _manhattan((r, g, b), (40, 62, 145)) <= 50:
        return True
    # Dark blue family (rounded panel) — exclude teal road markers (low R, mid G/B)
    if b > r + 38 and b > g + 28 and r < 115 and g < 115 and b > 105 and r > 5:
        if r < 25 and g > 55 and b > 55 and abs(g - b) < 25:
            return False
        return True
    return False


def _strip_horizontal_margins(img, margin_ratio: float = 0.88, tol: int = 165):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage

    img = img.convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    if w < 16 or h < 16:
        return img
    ref = QColor(img.pixel(0, h // 2))

    def column_margin_ratio(x: int) -> float:
        matched = 0
        for y in range(h):
            c = QColor(img.pixel(x, y))
            dist = (
                abs(c.red() - ref.red())
                + abs(c.green() - ref.green())
                + abs(c.blue() - ref.blue())
            )
            if dist <= tol:
                matched += 1
        return matched / float(h)

    left = 0
    while left < w and column_margin_ratio(left) > margin_ratio:
        left += 1
    right = w - 1
    while right >= left and column_margin_ratio(right) > margin_ratio:
        right -= 1
    new_w = right - left + 1
    if new_w < max(48, w // 6):
        return img
    return img.copy(QRect(left, 0, new_w, h))


def _isolate_content_on_black(img):
    """RGB32 in/out: background colors -> black; road, car, graph lines kept."""
    from PySide6.QtGui import QColor, QImage

    src = img.convertToFormat(QImage.Format.Format_RGB32)
    w, h = src.width(), src.height()
    out = QImage(w, h, QImage.Format.Format_RGB32)
    out.fill(0xFF000000)
    for y in range(h):
        for x in range(w):
            c = QColor(src.pixel(x, y))
            r, g, b = c.red(), c.green(), c.blue()
            if _is_background_rgb(r, g, b):
                continue
            out.setPixel(x, y, src.pixel(x, y))
    return out


def _bbox_non_black(img, thresh: int = 10):
    from PySide6.QtGui import QColor

    w, h = img.width(), img.height()
    min_x, min_y = w, h
    max_x, max_y = -1, -1
    for y in range(h):
        for x in range(w):
            c = QColor(img.pixel(x, y))
            if c.red() > thresh or c.green() > thresh or c.blue() > thresh:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < min_x:
        return None
    return min_x, min_y, max_x, max_y


def _to_square_centered_black(img):
    from PySide6.QtGui import QImage, QPainter

    bbox = _bbox_non_black(img)
    if bbox is None:
        return img
    min_x, min_y, max_x, max_y = bbox
    img = img.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    w, h = img.width(), img.height()
    side = max(w, h, 768)
    out = QImage(side, side, QImage.Format.Format_RGB32)
    out.fill(0xFF000000)
    ox = (side - w) // 2
    oy = (side - h) // 2
    p = QPainter(out)
    p.drawImage(ox, oy, img)
    p.end()
    return out


def main() -> int:
    if not _SRC.is_file():
        print(f"Missing source: {_SRC}", file=sys.stderr)
        return 1

    try:
        from PySide6.QtGui import QImage
    except ModuleNotFoundError:
        print(
            "PySide6 is required. Use the project interpreter, e.g.:\n"
            "  .venv/bin/python3 scripts/build_app_icon_square_rgba.py\n"
            "  uv run python3 scripts/build_app_icon_square_rgba.py",
            file=sys.stderr,
        )
        return 1

    img = QImage(str(_SRC))
    if img.isNull():
        print(f"Failed to load: {_SRC}", file=sys.stderr)
        return 1

    img = _strip_horizontal_margins(img)
    img = _isolate_content_on_black(img)
    img = _to_square_centered_black(img)

    _DST.parent.mkdir(parents=True, exist_ok=True)
    if not img.save(str(_DST), "PNG"):
        print(f"Failed to save: {_DST}", file=sys.stderr)
        return 1

    print(f"Wrote {img.width()}x{img.height()} RGB (black + content) -> {_DST}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT))
    raise SystemExit(main())
