#!/usr/bin/env python3
"""
Validate a step dataset output directory.

Checks:
  1. static.json exists and has non-empty junctions and road_edges
  2. Every step file has a matching label file (same timestamp)
  3. Every label file has a matching step file
  4. Vehicle IDs in step nodes == vehicle IDs in label labels (exact match)
  5. All step files have the required top-level keys
  6. All label files have the required top-level keys
  7. No vehicle appears in a label without an 'eta' field
  8. Prints a summary: file counts, vehicle count range, any mismatches

Usage:
    uv run python scripts/validate_steps.py --output /path/to/output
    uv run python scripts/validate_steps.py --output /path/to/output --verbose
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

STEP_KEYS = {"step", "nodes", "road_edges_dynamic", "dynamic_edges"}
LABEL_KEYS = {"timestamp", "labels"}


def _read_json(path: Path) -> Optional[Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None


def validate(output_dir: Path, verbose: bool) -> bool:
    ok = True

    # --- static.json ---
    static_path = output_dir / "static.json"
    if not static_path.exists():
        static_path = output_dir / "static.json.gz"
    if not static_path.exists():
        print("FAIL  static.json not found")
        ok = False
    else:
        static = _read_json(static_path)
        if static is None:
            print("FAIL  static.json could not be parsed")
            ok = False
        else:
            n_j = len(static.get("junctions", []))
            n_e = len(static.get("road_edges", []))
            if n_j == 0 or n_e == 0:
                print(f"FAIL  static.json: {n_j} junctions, {n_e} road_edges (expected > 0)")
                ok = False
            else:
                print(f"OK    static.json: {n_j} junctions, {n_e} road_edges")

    snapshots_dir = output_dir / "snapshots"
    labels_dir = output_dir / "labels"

    if not snapshots_dir.exists():
        print("FAIL  snapshots/ directory not found")
        return False
    if not labels_dir.exists():
        print("FAIL  labels/ directory not found")
        return False

    # Build index of step and label files by timestamp key
    step_files: Dict[str, Path] = {}
    for p in sorted(snapshots_dir.iterdir()):
        if p.name.startswith("step_") and (".json" in p.suffixes):
            key = p.name.replace("step_", "").replace(".json.gz", "").replace(".json", "")
            step_files[key] = p

    label_files: Dict[str, Path] = {}
    for p in sorted(labels_dir.iterdir()):
        if p.name.startswith("label_") and (".json" in p.suffixes):
            key = p.name.replace("label_", "").replace(".json.gz", "").replace(".json", "")
            label_files[key] = p

    if not step_files:
        print("FAIL  no step files found in snapshots/")
        return False

    print(f"\nFound {len(step_files)} step files, {len(label_files)} label files")

    # Check every step has a label and vice-versa
    step_keys = set(step_files)
    label_keys = set(label_files)
    missing_labels = step_keys - label_keys
    missing_steps = label_keys - step_keys

    if missing_labels:
        for k in sorted(missing_labels)[:5]:
            print(f"FAIL  step_{k} has no matching label_{k}")
        if len(missing_labels) > 5:
            print(f"      ... and {len(missing_labels) - 5} more")
        ok = False
    if missing_steps:
        for k in sorted(missing_steps)[:5]:
            print(f"FAIL  label_{k} has no matching step_{k}")
        if len(missing_steps) > 5:
            print(f"      ... and {len(missing_steps) - 5} more")
        ok = False

    # Per-file content checks
    mismatches = 0
    schema_errors = 0
    veh_counts = []
    common_keys = sorted(step_keys & label_keys)

    for key in common_keys:
        step = _read_json(step_files[key])
        label = _read_json(label_files[key])

        if step is None:
            print(f"FAIL  step_{key}: could not parse JSON")
            schema_errors += 1
            ok = False
            continue
        if label is None:
            print(f"FAIL  label_{key}: could not parse JSON")
            schema_errors += 1
            ok = False
            continue

        # Schema check
        missing_step_keys = STEP_KEYS - set(step.keys())
        missing_label_keys = LABEL_KEYS - set(label.keys())
        if missing_step_keys:
            print(f"FAIL  step_{key}: missing keys {missing_step_keys}")
            schema_errors += 1
            ok = False
        if missing_label_keys:
            print(f"FAIL  label_{key}: missing keys {missing_label_keys}")
            schema_errors += 1
            ok = False
        if missing_step_keys or missing_label_keys:
            continue

        # Vehicle ID match
        step_ids = {n["id"] for n in step["nodes"] if "id" in n}
        label_ids = {l["id"] for l in label["labels"] if "id" in l}

        veh_counts.append(len(step_ids))

        if step_ids != label_ids:
            mismatches += 1
            ok = False
            if verbose or mismatches <= 3:
                only_step = step_ids - label_ids
                only_label = label_ids - step_ids
                print(
                    f"FAIL  ts={key}: step has {len(step_ids)} vehicles, "
                    f"label has {len(label_ids)}  "
                    f"(step-only={len(only_step)}, label-only={len(only_label)})"
                )
                if verbose:
                    if only_step:
                        print(f"      step-only:  {sorted(only_step)[:5]}")
                    if only_label:
                        print(f"      label-only: {sorted(only_label)[:5]}")
        elif verbose:
            print(f"OK    ts={key}: {len(step_ids)} vehicles match")

        # ETA field check
        missing_eta = [l["id"] for l in label["labels"] if "eta" not in l]
        if missing_eta:
            print(f"FAIL  label_{key}: {len(missing_eta)} entries missing 'eta' field")
            ok = False

    # Summary
    print()
    if mismatches == 0 and schema_errors == 0 and not missing_labels and not missing_steps:
        print(f"PASS  All {len(common_keys)} snapshot pairs validated successfully.")
    else:
        print(
            f"FAIL  {mismatches} vehicle-set mismatches, "
            f"{schema_errors} schema errors, "
            f"{len(missing_labels)} missing labels, "
            f"{len(missing_steps)} missing steps."
        )

    if veh_counts:
        print(
            f"      Vehicle counts per snapshot: "
            f"min={min(veh_counts)}, max={max(veh_counts)}, "
            f"avg={sum(veh_counts)/len(veh_counts):.1f}"
        )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate a step dataset output directory.")
    parser.add_argument("--output", type=Path, required=True, help="Dataset output folder to validate")
    parser.add_argument("--verbose", action="store_true", help="Print per-file OK lines and mismatch details")
    args = parser.parse_args()

    if not args.output.exists():
        print(f"Output directory not found: {args.output}", file=sys.stderr)
        sys.exit(1)

    passed = validate(args.output, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
