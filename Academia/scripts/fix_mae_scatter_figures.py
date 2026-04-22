#!/usr/bin/env python3
"""
Post-process MAE scatter PNGs for the NiDS paper: pure white background, paper title band,
and a rebuilt x-axis (spine, ticks 0--80, label).

Pipeline:
  1. Find the x-axis row y_axis (bottom strip): the row with the longest contiguous
     dark segment (matplotlib spine ink).
  2. On that row, take that segment's endpoints (x0, x1): left and right plot border
     where the horizontal axis meets the frame (0 and 80 minutes).
  3. Tick x positions: linear spacing x0 + k * (x1 - x0) / 8 for k = 0..8, with
     k = 0 and k = 8 pinned exactly to x0 and x1 so the axis uses the full span.
     (Vertical edge aids are available in code for future tuning; linear spacing
     matches the matplotlib grid when x0,x1 come from the spine segment.)
  4. Clear only pixels with x0 <= x <= y_axis band (never x < x0), so the y-axis
     ``0`` (MAE) to the left of the plot is not painted over.
  5. Redraw spine, ticks, 0..80 labels, and ``Trip Duration (minutes)``.

Run from repo root:
  python3 Academia/scripts/fix_mae_scatter_figures.py
  python3 Academia/scripts/fix_mae_scatter_figures.py --repair-xaxis
"""
from __future__ import annotations

import argparse
import os
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")

FILES = (
    ("sim_mae_scatter.png", "Trip duration vs. MAE (simulation)"),
    ("con_mae_scatter.png", "Trip duration vs. MAE (trajectory conversion)"),
)

X_LABEL = "Trip Duration (minutes)"
TICK_VALUES = (0, 10, 20, 30, 40, 50, 60, 70, 80)

TOP_PAD = 48
BOTTOM_PAD = 40
PROCESSED_HEIGHT = 700 + TOP_PAD + BOTTOM_PAD

CONTENT_TOP = TOP_PAD
AXIS_INK_SUM_MAX = 130
MIN_AXIS_RUN = 300
# Clear slightly past the right plot border so the old ``80`` glyph is removed.
# Keep the left edge exact at x0 so the y-axis ``0`` label is not erased.
CLEAR_RIGHT_PAD = 28
# Plot body for vertical-edge aid (below title, above x-axis clutter)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _longest_dark_run_on_row(row: list[int], thresh: int) -> tuple[int, int, int]:
    """Return (length, x_start, x_end) for longest contiguous run with sum < thresh."""
    best = (0, -1, -1)
    cur_s: int | None = None
    for x, s in enumerate(row):
        if s < thresh:
            if cur_s is None:
                cur_s = x
        else:
            if cur_s is not None:
                ln = x - cur_s
                if ln > best[0]:
                    best = (ln, cur_s, x - 1)
                cur_s = None
    if cur_s is not None:
        ln = len(row) - cur_s
        if ln > best[0]:
            best = (ln, cur_s, len(row) - 1)
    return best[0], best[1], best[2]


def detect_y_axis_and_spine_extent(im: Image.Image) -> tuple[int, int, int]:
    """
    Find axis row y_axis and spine endpoints (x0, x1) from the longest dark horizontal
    segment on that row (full axis from minute 0 to minute 80).
    """
    im = im.convert("RGB")
    px = im.load()
    w, h = im.size
    y_lo = max(CONTENT_TOP + 320, h - 220)
    y_hi = h - 12
    thresh = AXIS_INK_SUM_MAX * 3

    best_y = -1
    best_len = 0
    best_seg = (-1, -1)

    for y in range(y_lo, y_hi):
        row = [sum(px[x, y]) for x in range(w)]
        ln, xs, xe = _longest_dark_run_on_row(row, thresh)
        if ln > best_len:
            best_len = ln
            best_y = y
            best_seg = (xs, xe)

    if best_len < MIN_AXIS_RUN or best_y < 0:
        raise RuntimeError(
            f"Could not detect x-axis spine (best run length {best_len}). "
            "Expected a long dark horizontal segment in the lower plot area."
        )
    x0, x1 = best_seg
    return best_y, x0, x1


def tick_x_positions_linear(x0: int, x1: int) -> list[tuple[int, int]]:
    """Pairs (value_minutes, x_pixel); ends pinned to spine endpoints."""
    span = x1 - x0
    out: list[tuple[int, int]] = []
    for v in TICK_VALUES:
        k = v // 10
        xv = int(round(x0 + (k / 8.0) * span))
        out.append((v, xv))
    return out


def clear_below_axis(
    im: Image.Image,
    y_axis: int,
    x0: int,
    x1: int,
) -> None:
    """
    White-fill from the x-axis row through the bottom, starting exactly at x0 and
    extending slightly past x1. This preserves the y-axis MAE ``0`` at the left while
    removing stale right-edge pixels around the ``80`` tick label.
    """
    px = im.load()
    w, h = im.size
    xa = max(0, x0)
    xb = min(w - 1, x1 + CLEAR_RIGHT_PAD)
    for y in range(y_axis, h):
        for x in range(xa, xb + 1):
            px[x, y] = (255, 255, 255)


def redraw_x_axis_infrastructure(
    im: Image.Image,
    y_axis: int,
    x0: int,
    x1: int,
    ticks: list[tuple[int, int]],
) -> None:
    draw = ImageDraw.Draw(im)
    ink = (33, 33, 33)
    w, h = im.size

    draw.line([(x0, y_axis), (x1, y_axis)], fill=ink, width=1)

    tick_len = 5
    font_tick = _load_font(12)
    font_label = _load_font(15)

    for v, xc in ticks:
        draw.line([(xc, y_axis), (xc, y_axis + tick_len)], fill=ink, width=1)
        ty = y_axis + tick_len + 3
        draw.text((xc, ty), str(v), fill=ink, font=font_tick, anchor="mt")

    bbox = draw.textbbox((0, 0), X_LABEL, font=font_label)
    label_h = bbox[3] - bbox[1]
    gap = 10
    label_y = min(
        h - 8 - label_h // 2,
        y_axis + tick_len + 22 + gap,
    )
    draw.text((w // 2, label_y), X_LABEL, fill=ink, font=font_label, anchor="mm")


def rebuild_x_axis_bottom(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    y_axis, x0, x1 = detect_y_axis_and_spine_extent(im)
    print(f"  spine: y={y_axis}, x0={x0}, x1={x1}, span={x1 - x0}")

    ticks = tick_x_positions_linear(x0, x1)
    print("  ticks:", ", ".join(f"{v}@{x}" for v, x in ticks))

    clear_below_axis(im, y_axis, x0, x1)
    redraw_x_axis_infrastructure(im, y_axis, x0, x1, ticks)
    return im


def whiten_background(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    px = im.load()
    w, h = im.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            lo, hi = min(r, g, b), max(r, g, b)
            if hi < 200:
                continue
            if hi - lo > 38:
                continue
            if r >= 228 and g >= 228 and b >= 228:
                px[x, y] = (255, 255, 255)
    return im


def add_title_and_padding(im: Image.Image, title: str) -> Image.Image:
    w, h = im.size
    nw = w
    nh = h + TOP_PAD + BOTTOM_PAD
    out = Image.new("RGB", (nw, nh), (255, 255, 255))
    out.paste(im, (0, TOP_PAD))
    draw = ImageDraw.Draw(out)
    font = _load_font(17)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (nw - tw) // 2
    ty = max(6, (TOP_PAD - th) // 2)
    draw.text((tx, ty), title, fill=(33, 33, 33), font=font)
    return out


def repair_file(path: str, filename: str) -> None:
    im = Image.open(path)
    if im.size[1] != PROCESSED_HEIGHT:
        print(f"Skip repair {filename}: expected height {PROCESSED_HEIGHT}, got {im.size[1]}")
        return
    im = rebuild_x_axis_bottom(im)
    im.save(path, format="PNG", optimize=True)
    print(f"Rebuilt x-axis: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix MAE scatter PNGs for the paper.")
    parser.add_argument(
        "--repair-xaxis",
        action="store_true",
        help="Only rebuild x-axis (1024×788 PNGs).",
    )
    args = parser.parse_args()

    images_dir = os.path.normpath(IMAGES_DIR)
    if args.repair_xaxis:
        for filename, _ in FILES:
            path = os.path.join(images_dir, filename)
            if not os.path.isfile(path):
                raise SystemExit(f"Missing {path}")
            repair_file(path, filename)
        return

    for filename, title in FILES:
        path = os.path.join(images_dir, filename)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {path}")
        im = Image.open(path)
        if im.size[1] == PROCESSED_HEIGHT:
            print(f"Skip {filename}: already processed ({im.size[0]}x{im.size[1]})")
            continue
        im = whiten_background(im)
        im = add_title_and_padding(im, title)
        im.save(path, format="PNG", optimize=True)
        print(f"Wrote {path} ({im.size[0]}x{im.size[1]})")
        repair_file(path, filename)


if __name__ == "__main__":
    main()
