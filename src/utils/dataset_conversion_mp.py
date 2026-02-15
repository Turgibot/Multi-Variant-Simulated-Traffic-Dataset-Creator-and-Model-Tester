"""
Multiprocessing helpers for dataset conversion.
Used by the app's DatasetGenerationWorker - no Qt imports to keep worker processes lightweight.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from src.utils.network_parser import NetworkParser
from src.utils.route_finding import build_edges_data, build_node_positions
from src.utils.trajectory_converter import convert_trajectory

_worker_state: Optional[Tuple] = None


def init_worker(
    net_path: str,
    out_path: str,
    use_polygon: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> None:
    """Initialize worker process. Called once per worker."""
    global _worker_state
    np_local = NetworkParser(net_path)
    conv = np_local.conv_boundary
    bounds = np_local.get_bounds()
    y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
    y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)
    edges_data = build_edges_data(np_local)
    edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
    node_positions = build_node_positions(np_local)
    _worker_state = (
        np_local,
        edges_data,
        edge_shapes,
        node_positions,
        y_min,
        y_max,
        out_path,
        use_polygon,
        offset_x,
        offset_y,
    )


def process_one_trajectory(
    task: Tuple[int, List, Optional[int]],
) -> Tuple[int, int]:
    """Process a single trajectory. Returns (saved, 1)."""
    global _worker_state
    trip_num, polyline, base_ts = task
    (
        np_local,
        edges_data,
        edge_shapes,
        node_positions,
        y_min,
        y_max,
        out_path,
        use_polygon,
        offset_x,
        offset_y,
    ) = _worker_state
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
        offset_x=offset_x,
        offset_y=offset_y,
    )
    if rec:
        out_file = Path(out_path) / f"traj_{trip_num}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        return 1, 1
    return 0, 1


def run_multiprocess(
    trajectories: List[Tuple[int, List, Optional[int]]],
    network_path: str,
    output_path: str,
    workers: int,
    use_polygon: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancelled_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int]:
    """
    Run conversion with multiprocessing.
    progress_callback(current, total) is called as results arrive.
    cancelled_callback() is checked between batches (not per-item).
    Returns (saved_count, total_processed).
    """
    from multiprocessing import Pool

    os.makedirs(output_path, exist_ok=True)
    total = len(trajectories)
    if total == 0:
        return 0, 0

    saved_count = 0
    total_processed = 0
    chunk_size = max(1, total // (workers * 4))  # reasonable chunks for progress

    with Pool(
        workers,
        initializer=init_worker,
        initargs=(network_path, output_path, use_polygon, offset_x, offset_y),
    ) as pool:
        for i, result in enumerate(pool.imap(process_one_trajectory, trajectories, chunksize=chunk_size)):
            if cancelled_callback and cancelled_callback():
                break
            s, t = result
            saved_count += s
            total_processed += t
            if progress_callback:
                progress_callback(total_processed, total)

    return saved_count, total_processed
