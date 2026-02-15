#!/usr/bin/env python3
"""
Optimized standalone script to convert train.csv trajectories to JSON (SUMO routes).
Same logic and results as the GUI DatasetConversionPage, but significantly faster.

Optimizations:
- Single-pass CSV reading (no re-reading from start for each trajectory)
- Spatial index filtering for edge matching (reduces O(edges) to O(nearby_edges))
- Optional multiprocessing for parallel trajectory conversion
- Pre-built caches (edges_data, node_positions, spatial_index)

Usage:
  python scripts/convert_trajectories_fast.py \\
    --train Porto/dataset/train.csv \\
    --network Porto/config/porto.net.xml \\
    --output output_dir \\
    --start 1 --end 1000 --workers 4

  (omit --start/--end/--workers to use defaults: 1, 1000, 1)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.utils.network_parser import NetworkParser
from src.utils.route_finding import build_edges_data, build_node_positions
from src.utils.trajectory_converter import (
    convert_trajectory,
    iter_trajectories_from_csv,
)


def run_single_process(
    train_path: str,
    network_path: str,
    output_path: str,
    start_traj: int,
    last_traj: int,
    use_polygon: bool = False,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Run conversion in single process. Returns (saved_count, total_processed)."""
    os.makedirs(output_path, exist_ok=True)

    if verbose:
        print("Loading network...")
    network_parser = NetworkParser(network_path)
    conv = network_parser.conv_boundary
    bounds = network_parser.get_bounds()
    y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
    y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)

    if verbose:
        print("Building edges data...")
    edges_data = build_edges_data(network_parser)
    edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
    node_positions = build_node_positions(network_parser)

    saved_count = 0
    total_processed = 0
    total = max(0, last_traj - start_traj + 1)

    iterator = iter_trajectories_from_csv(train_path, start_traj, last_traj)
    if verbose:
        iterator = tqdm(iterator, total=total, desc="Converting", unit="traj")

    for trip_num, polyline, base_timestamp in iterator:
        total_processed += 1

        rec = convert_trajectory(
            trip_num,
            polyline,
            base_timestamp,
            network_parser,
            edges_data,
            edge_shapes,
            node_positions,
            y_min,
            y_max,
            use_polygon=use_polygon,
        )
        if rec:
            out_file = Path(output_path) / f"traj_{trip_num}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            saved_count += 1

    return saved_count, total_processed


# Module-level state for worker processes (set by initializer)
_worker_state: Optional[Tuple] = None


def _init_worker(net_path: str, out_path: str, use_polygon: bool = False) -> None:
    """Initialize worker with network (called once per worker process)."""
    global _worker_state
    np_local = NetworkParser(net_path)
    conv = np_local.conv_boundary
    bounds = np_local.get_bounds()
    y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
    y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)
    edges_data = build_edges_data(np_local)
    edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
    node_positions = build_node_positions(np_local)
    _worker_state = (np_local, edges_data, edge_shapes, node_positions, y_min, y_max, out_path, use_polygon)


def _process_one_trajectory(task: Tuple[int, List, Optional[int]]) -> Tuple[int, int]:
    """Process a single trajectory. Worker must be initialized. Returns (saved, 1)."""
    global _worker_state
    trip_num, polyline, base_ts = task
    np_local, edges_data, edge_shapes, node_positions, y_min, y_max, out_path, use_polygon = _worker_state
    rec = convert_trajectory(
        trip_num,
        polyline,
        base_ts,
        np_local,
        edges_data,
        edge_shapes,
        node_positions,
        y_min,
        y_max,
        use_polygon=use_polygon,
    )
    if rec:
        out_file = Path(out_path) / f"traj_{trip_num}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        return 1, 1
    return 0, 1


def run_multiprocess(
    train_path: str,
    network_path: str,
    output_path: str,
    start_traj: int,
    last_traj: int,
    workers: int,
    use_polygon: bool = False,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Run conversion with multiprocessing. Returns (saved_count, total_processed)."""
    from multiprocessing import Pool

    os.makedirs(output_path, exist_ok=True)

    # Collect all trajectories in memory (needed for parallel distribution)
    if verbose:
        print("Loading trajectories from CSV (single pass)...")
    trajectories = list(iter_trajectories_from_csv(train_path, start_traj, last_traj))
    total = len(trajectories)
    if total == 0:
        return 0, 0

    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(network_path, output_path, use_polygon),
    ) as pool:
        imap = pool.imap(_process_one_trajectory, trajectories)
        if verbose:
            imap = tqdm(imap, total=total, desc="Converting", unit="traj")
        results = list(imap)

    saved_count = sum(r[0] for r in results)
    total_processed = sum(r[1] for r in results)
    return saved_count, total_processed


def _run_comparison(args: argparse.Namespace) -> None:
    """Run conversion with and without polygon, compare saved counts."""
    import time as _time
    base = Path(args.output)
    out_no = str(base / "no_polygon")
    out_yes = str(base / "with_polygon")
    base.mkdir(parents=True, exist_ok=True)
    train_path = str(Path(args.train).resolve())
    network_path = str(Path(args.network).resolve())
    start_traj = max(1, args.start)
    last_traj = max(start_traj, args.end)
    workers = max(1, args.workers)
    verbose = not args.quiet

    print(f"Comparing trajectories {start_traj}-{last_traj}")
    print("=" * 50)

    t0 = _time.perf_counter()
    if workers == 1:
        saved_no, _ = run_single_process(train_path, network_path, out_no, start_traj, last_traj, use_polygon=False, verbose=verbose)
    else:
        saved_no, _ = run_multiprocess(train_path, network_path, out_no, start_traj, last_traj, workers, use_polygon=False, verbose=verbose)
    elapsed_no = _time.perf_counter() - t0
    print(f"\nWithout polygon: {saved_no} saved in {elapsed_no:.1f}s")

    t0 = _time.perf_counter()
    if workers == 1:
        saved_yes, _ = run_single_process(train_path, network_path, out_yes, start_traj, last_traj, use_polygon=True, verbose=verbose)
    else:
        saved_yes, _ = run_multiprocess(train_path, network_path, out_yes, start_traj, last_traj, workers, use_polygon=True, verbose=verbose)
    elapsed_yes = _time.perf_counter() - t0
    print(f"With polygon:    {saved_yes} saved in {elapsed_yes:.1f}s")

    print("=" * 50)
    diff = saved_no - saved_yes
    if diff > 0:
        print(f"Polygon misses {diff} trajectories (use full network for max coverage)")
    elif diff < 0:
        print(f"Polygon finds {-diff} more (edge case)")
    else:
        print("Same count (may differ in which trajectories)")
    only_no = set(int(f.stem.split("_")[1]) for f in Path(out_no).glob("traj_*.json"))
    only_yes = set(int(f.stem.split("_")[1]) for f in Path(out_yes).glob("traj_*.json"))
    only_in_no = only_no - only_yes
    only_in_yes = only_yes - only_no
    if only_in_no or only_in_yes:
        if only_in_no:
            print(f"Only without polygon: {sorted(only_in_no)[:10]}{'...' if len(only_in_no) > 10 else ''}")
        if only_in_yes:
            print(f"Only with polygon:    {sorted(only_in_yes)[:10]}{'...' if len(only_in_yes) > 10 else ''}")
    speedup = elapsed_no / elapsed_yes if elapsed_yes > 0 else 0
    print(f"Speed: polygon {'%.1fx faster' % speedup if speedup > 1 else '%.1fx slower' % (1/speedup) if speedup < 1 else 'same'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert train.csv trajectories to JSON (SUMO routes) - optimized standalone script"
    )
    parser.add_argument("--train", "-t", required=True, help="Path to train.csv")
    parser.add_argument("--network", "-n", required=True, help="Path to .net.xml")
    parser.add_argument("--output", "-o", required=True, help="Output directory for JSON files")
    parser.add_argument("--start", "-s", type=int, default=1, help="First trajectory index (1-based)")
    parser.add_argument("--end", "-e", type=int, default=1000, help="Last trajectory index (inclusive)")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (1 = single process)")
    parser.add_argument("--use-polygon", action="store_true", help="Restrict Dijkstra to edges inside trajectory polygon (faster, may miss some routes)")
    parser.add_argument("--compare", action="store_true", help="Run both modes and compare saved counts (uses --output as base dir)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if args.compare:
        _run_comparison(args)
        return

    train_path = Path(args.train)
    network_path = Path(args.network)
    if not train_path.exists():
        print(f"Error: train.csv not found: {train_path}")
        sys.exit(1)
    if not network_path.exists():
        print(f"Error: network file not found: {network_path}")
        sys.exit(1)

    start_traj = max(1, args.start)
    last_traj = max(start_traj, args.end)
    workers = max(1, args.workers)
    verbose = not args.quiet

    if verbose:
        print(f"Converting trajectories {start_traj}-{last_traj} from {train_path}")
        print(f"Network: {network_path}")
        print(f"Output: {args.output}")
        if workers > 1:
            print(f"Workers: {workers}")

    import time
    t0 = time.perf_counter()

    use_polygon = getattr(args, "use_polygon", False)
    if workers == 1:
        saved, total = run_single_process(
            str(train_path),
            str(network_path),
            args.output,
            start_traj,
            last_traj,
            use_polygon=use_polygon,
            verbose=verbose,
        )
    else:
        saved, total = run_multiprocess(
            str(train_path),
            str(network_path),
            args.output,
            start_traj,
            last_traj,
            workers,
            use_polygon=use_polygon,
            verbose=verbose,
        )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\nDone: {saved}/{total} trajectories saved in {elapsed:.1f}s ({elapsed/max(1,total):.2f}s per trajectory)")


if __name__ == "__main__":
    main()
