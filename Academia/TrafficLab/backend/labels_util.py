"""
Build backend/labels.json from labels_1.json + labels_2.json + labels_3.json when labels.json is absent.
Each part must be a JSON array of journey objects (same schema as the historical single file).
"""

from __future__ import annotations

import json
import os
from typing import Optional

PART_NAMES = ("labels_1.json", "labels_2.json", "labels_3.json")


def ensure_unified_labels_json(backend_dir: str) -> Optional[str]:
    """
    If labels.json already exists, return its path.
    Else if all labels_1/2/3.json exist, merge arrays and write labels.json atomically, then return path.
    Otherwise return None (callers may fall back to labels.json.example or seed errors).
    """
    labels_path = os.path.join(backend_dir, "labels.json")
    if os.path.isfile(labels_path):
        return labels_path

    part_paths = [os.path.join(backend_dir, name) for name in PART_NAMES]
    if not all(os.path.isfile(p) for p in part_paths):
        missing = [os.path.basename(p) for p in part_paths if not os.path.isfile(p)]
        print(
            f"ℹ️  No labels.json yet; split files missing ({', '.join(missing)}). "
            "Skipping merge."
        )
        return None

    print(
        "📦 Building labels.json from labels_1.json + labels_2.json + labels_3.json "
        "(first run; large files — this may take several minutes)..."
    )
    merged: list = []
    for p in part_paths:
        with open(p, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        if not isinstance(chunk, list):
            raise ValueError(f"{p} must contain a JSON array")
        merged.extend(chunk)
        print(f"   … loaded {os.path.basename(p)} (+{len(chunk)} rows, total {len(merged)})")

    tmp_path = labels_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        os.replace(tmp_path, labels_path)
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    print(f"✅ Wrote {labels_path} ({len(merged)} records)")
    return labels_path
