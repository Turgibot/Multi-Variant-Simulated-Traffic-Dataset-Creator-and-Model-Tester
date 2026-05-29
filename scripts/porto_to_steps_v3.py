#!/usr/bin/env python3
"""
Porto parquet/CSV → Step (Snapshot) Files  — v2 pipeline.

Improvements over csv_to_steps.py:
  - Parquet input (pandas): reads train/val/test.parquet directly
  - Step/label vehicle-set match guaranteed: only vehicles with a known
    destination timestamp appear in either file
  - Parallel snapshot writing: writer process pool instead of one thread
  - Higher throughput: imap chunksize 2 → 32
"""

import argparse
import ast
import csv
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sortedcontainers import SortedDict as _SortedDict
    _USE_SORTED_DICT = True
except ImportError:
    _SortedDict = None
    _USE_SORTED_DICT = False

from src.utils.route_finding import (
    EdgeSpatialIndex,
    build_base_adjacency,
    build_edge_shape_arrays,
    build_edges_data,
    build_node_positions,
    project_point_onto_polyline_with_segment_and_t,
)
from src.utils.trajectory_converter import _parse_csv_row, convert_trajectory
from src.utils.traffic_db import TrafficDB

# ---------------------------------------------------------------------------
# Multiprocessing worker state (trajectory conversion)
# ---------------------------------------------------------------------------

_mp_worker_state: Optional[Tuple] = None


def _mp_init_worker(net_path: str) -> None:
    global _mp_worker_state
    from src.utils.network_parser import NetworkParser

    np_local = NetworkParser(net_path)
    conv = np_local.conv_boundary
    bounds = np_local.get_bounds()
    y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
    y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)
    edges_data = build_edges_data(np_local)
    edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
    node_positions = build_node_positions(np_local)
    spatial_index = EdgeSpatialIndex(edges_data, cell_size=500.0)
    base_adj = build_base_adjacency(np_local)          # Opt 1: pre-built adjacency
    _mp_worker_state = (
        np_local, edges_data, edge_shapes, node_positions,
        y_min, y_max, spatial_index, base_adj,
    )


def _mp_convert_one(
    task: Tuple[int, List, Optional[int]],
) -> Optional[Tuple[int, List, Optional[int], Any]]:
    global _mp_worker_state
    if _mp_worker_state is None:
        return None
    trip_num, polyline, ts = task
    if ts is None:
        return None
    np_local, edges_data, edge_shapes, node_positions, y_min, y_max, spatial_index, base_adj = _mp_worker_state
    rec = convert_trajectory(
        trip_num, polyline, ts, np_local, edges_data, edge_shapes,
        node_positions, y_min, y_max, use_polygon=False, spatial_index=spatial_index,
        base_adj=base_adj,
    )
    if rec:
        return (trip_num, polyline, ts, rec)
    return None


# ---------------------------------------------------------------------------
# Snapshot writer (top-level so it's picklable by the writer pool)
# ---------------------------------------------------------------------------

def _write_snapshot(
    snapshots_dir: str,
    labels_dir: str,
    ts_val: int,
    step_data: Dict[str, Any],
    label_data: Dict[str, Any],
    compress: bool,
) -> None:
    nodes = step_data.get("nodes", [])
    if not nodes:
        print(f"skipping empty step at ts {ts_val} (no vehicles)", flush=True)
        return
    if compress:
        with gzip.open(f"{snapshots_dir}/step_{ts_val:012d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(step_data, f, separators=(",", ":"))
        with gzip.open(f"{labels_dir}/label_{ts_val:012d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(label_data, f, separators=(",", ":"))
    else:
        with open(f"{snapshots_dir}/step_{ts_val:012d}.json", "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2)
        with open(f"{labels_dir}/label_{ts_val:012d}.json", "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2)
    n_veh = len(nodes)
    ids = [str(n.get("id", "")) for n in nodes if n.get("id")]
    ids_str = ", ".join(ids[:40]) + (", ..." if len(ids) > 40 else "")
    print(f"creating ds files of ts {ts_val} with {n_veh} vehicles : {ids_str}", flush=True)


# ---------------------------------------------------------------------------
# Input streaming
# ---------------------------------------------------------------------------

def _stream_parquet_rows(parquet_path: Path) -> Iterator[Tuple[int, List, int]]:
    """
    Yield (row_num, polyline_list, timestamp) from a .parquet file.
    Skips rows with missing_data=True or unparseable polylines.
    Validates that timestamps are non-decreasing.
    """
    import pandas as pd

    df = pd.read_parquet(str(parquet_path), columns=["timestamp", "polyline", "missing_data"])
    last_ts: Optional[int] = None
    for row_num, row in enumerate(df.itertuples(index=False), 1):
        if row.missing_data:
            continue
        try:
            polyline = ast.literal_eval(row.polyline)
        except (ValueError, SyntaxError):
            continue
        if not polyline or len(polyline) < 2:
            continue
        ts = int(row.timestamp)
        if last_ts is not None and ts < last_ts:
            print(
                f"Warning: parquet not sorted by timestamp at row {row_num} "
                f"(ts={ts} < prev={last_ts}). Rows may arrive out of order.",
                file=sys.stderr,
            )
        last_ts = ts
        yield row_num, polyline, ts


def _stream_csv_rows_simple(
    csv_path: Path,
) -> Iterator[Tuple[int, List, Optional[int]]]:
    """Stream (row_num, polyline, timestamp) from a pre-sorted CSV."""
    ts_idx: Optional[int] = None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            for i, h in enumerate(header):
                if str(h).strip('"') == "TIMESTAMP":
                    ts_idx = i
                    break
        for row_num, row in enumerate(reader, 1):
            polyline, timestamp = _parse_csv_row(row, ts_idx)
            if polyline:
                yield row_num, polyline, timestamp


def stream_input(input_path: Path) -> Iterator[Tuple[int, List, Optional[int]]]:
    """Auto-detect parquet vs CSV and return a row stream."""
    if input_path.suffix.lower() == ".parquet":
        return _stream_parquet_rows(input_path)
    return _stream_csv_rows_simple(input_path)


# ---------------------------------------------------------------------------
# Network / graph builders  (identical to csv_to_steps.py)
# ---------------------------------------------------------------------------

def _strip_lane_suffix(edge_id: str) -> str:
    return edge_id.split("#")[0] if "#" in edge_id else edge_id


def build_junctions_map(net: Any) -> Dict[str, Dict[str, Any]]:
    junctions: Dict[str, Dict[str, Any]] = {}
    for node in net.getNodes():
        nid = node.getID()
        if nid.startswith(":"):
            continue
        coord = node.getCoord()
        junctions[nid] = {
            "id": nid,
            "x": coord[0],
            "y": coord[1],
            "type": node.getType() or "priority",
            "zone": "",
            "incoming": [e.getID() for e in node.getIncoming()],
            "outgoing": [e.getID() for e in node.getOutgoing()],
        }
    return junctions


def build_edges_map(net: Any) -> Dict[str, Dict[str, Any]]:
    by_base: Dict[str, List[Any]] = {}
    for edge in net.getEdges():
        eid = edge.getID()
        if edge.getFunction() == "internal":
            continue
        base_id = _strip_lane_suffix(eid)
        by_base.setdefault(base_id, []).append(edge)
    edges: Dict[str, Dict[str, Any]] = {}
    for base_id, lane_edges in by_base.items():
        first = lane_edges[0]
        from_node = first.getFromNode()
        to_node = first.getToNode()
        edges[base_id] = {
            "id": base_id,
            "from": from_node.getID() if from_node else "",
            "to": to_node.getID() if to_node else "",
            "edge_type": 0,
            "speed": first.getSpeed(),
            "length": first.getLength(),
            "num_lanes": sum(e.getLaneNumber() for e in lane_edges),
            "zone": "",
            "density": 0.0,
            "avg_speed": 0.0,
            "edge_demand": 0.0,
            "vehicles_on_road": [],
        }
    return edges


def create_vehicle_node(vehicle_id: str) -> Dict[str, Any]:
    return {
        "id": vehicle_id,
        "vehicle_type": "passenger",
        "length": 0.0,
        "width": 0.0,
        "height": 0.0,
        "speed": 0.0,
        "acceleration": 0.0,
        "current_x": 0.0,
        "current_y": 0.0,
        "current_zone": "",
        "current_edge": "",
        "current_position": 0.0,
        "origin_name": "",
        "origin_zone": "",
        "origin_edge": "",
        "origin_position": 0.0,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "origin_start_sec": 0,
        "route": [],
        "route_length": 0.0,
        "route_left": [],
        "route_length_left": 0.0,
        "destination_name": "",
        "destination_edge": "",
        "destination_position": 0.0,
        "destination_x": 0.0,
        "destination_y": 0.0,
    }


def _build_vehicle_info(
    vehicle_id: str,
    speed: float,
    acceleration: float,
    current_x: float,
    current_y: float,
    current_zone: str,
    current_edge: str,
    current_position: float,
    route_left: List[str],
    route_length_left: float,
) -> Dict[str, Any]:
    return {
        "id": vehicle_id,
        "speed": speed,
        "acceleration": acceleration,
        "current_x": current_x,
        "current_y": current_y,
        "current_zone": current_zone,
        "current_edge": current_edge,
        "current_position": current_position,
        "route_left": route_left,
        "route_length_left": route_length_left,
    }


def _get_edge_shape(edge_id: str, edge_shapes: Dict[str, List]) -> Optional[List]:
    if edge_id in edge_shapes:
        return edge_shapes[edge_id]
    for eid, shape in edge_shapes.items():
        if "#" in eid and eid.split("#")[0] == edge_id:
            return shape
    if "#" not in edge_id and f"{edge_id}#0" in edge_shapes:
        return edge_shapes[f"{edge_id}#0"]
    return None


def _position_on_edge_from_coords(
    x: float, y: float, edge_id: str, edge_shapes: Dict[str, List]
) -> Optional[float]:
    shape = _get_edge_shape(edge_id, edge_shapes)
    if not shape or len(shape) < 2:
        return None
    (_, _), seg_idx, t = project_point_onto_polyline_with_segment_and_t(x, y, shape)
    dist = 0.0
    for i in range(len(shape) - 1):
        x1, y1 = shape[i][0], shape[i][1]
        x2, y2 = shape[i + 1][0], shape[i + 1][1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if i < seg_idx:
            dist += seg_len
        elif i == seg_idx:
            dist += t * seg_len
            break
    return dist


def _dyn_edge_id(from_id: str, to_id: str) -> str:
    sig = f"{from_id}_{to_id}"
    h = int(hashlib.md5(sig.encode()).hexdigest(), 16) % (10 ** 9)
    return f"dyn_{h:09d}"


def create_dynamic_edges(
    edges: Dict[str, Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    dynamic: List[Dict[str, Any]] = []
    for edge_id, edge_data in edges.items():
        veh_ids = edge_data.get("vehicles_on_road", [])
        if not veh_ids:
            continue
        junction_from = edge_data.get("from", "")
        junction_to = edge_data.get("to", "")
        sorted_vehs = sorted(
            (vid for vid in veh_ids if vid in vehicles),
            key=lambda vid: vehicles[vid].get("current_position", 0.0),
        )
        if not sorted_vehs:
            continue
        prev = junction_from
        for i, vid in enumerate(sorted_vehs):
            edge_type = 1 if i == 0 else 2
            dyn_id = _dyn_edge_id(prev, vid)
            dynamic.append({
                "id": dyn_id, "from": prev, "to": vid, "edge_type": edge_type,
                "speed": 0.0, "length": 0.0, "num_lanes": 0, "zone": "",
                "density": 0.0, "avg_speed": 0.0, "vehicles_on_road": [],
            })
            prev = vid
        dyn_id = _dyn_edge_id(prev, junction_to)
        dynamic.append({
            "id": dyn_id, "from": prev, "to": junction_to, "edge_type": 3,
            "speed": 0.0, "length": 0.0, "num_lanes": 0, "zone": "",
            "density": 0.0, "avg_speed": 0.0, "vehicles_on_road": [],
        })
    return dynamic


ACCELERATION_INTERVAL_SEC = 15.0
EDGE_DEMAND_TAU_SEC = 600.0
STATIC_EDGE_EXCLUDE = {"vehicles_on_road", "edge_demand", "avg_speed", "density"}


def build_static_json(db: TrafficDB) -> Dict[str, Any]:
    junctions = list(db.junctions.values())
    road_edges_static = [
        {k: v for k, v in e.items() if k not in STATIC_EDGE_EXCLUDE}
        for e in db.road_edges.values()
    ]
    return {"junctions": junctions, "road_edges": road_edges_static}


def build_step_json(
    db: TrafficDB,
    snapshot_timestamp: int,
    valid_vids: FrozenSet[str],
) -> Dict[str, Any]:
    """Build step snapshot. Only nodes in valid_vids are included."""
    valid_vehicles = {vid: v for vid, v in db.vehicles.items() if vid in valid_vids}
    dynamic_edges = create_dynamic_edges(db.road_edges, valid_vehicles)
    road_edges_dynamic = [
        {
            "id": e["id"],
            "vehicles_on_road": [v for v in e.get("vehicles_on_road", []) if v in valid_vids],
            "edge_demand": e.get("edge_demand", 0.0),
            "avg_speed": e.get("avg_speed", 0.0),
            "density": e.get("density", 0.0),
        }
        for e in db.road_edges.values()
        if e.get("vehicles_on_road") or e.get("edge_demand", 0.0) or
           e.get("avg_speed", 0.0) or e.get("density", 0.0)
    ]
    return {
        "step": snapshot_timestamp,
        "nodes": list(valid_vehicles.values()),
        "road_edges_dynamic": road_edges_dynamic,
        "dynamic_edges": dynamic_edges,
    }


def build_label_json(
    db: TrafficDB,
    snapshot_timestamp: int,
    vehicle_seg_last_ts: Dict[str, int],
    valid_vids: FrozenSet[str],
) -> Dict[str, Any]:
    """Build label JSON. Only vehicles in valid_vids are included."""
    labels = []
    for vid in valid_vids:
        dest_ts = vehicle_seg_last_ts.get(vid)
        if dest_ts is not None:
            eta = max(0, dest_ts - snapshot_timestamp)
            labels.append({"id": vid, "eta": eta})
    return {"timestamp": snapshot_timestamp, "labels": labels}


def update_db_from_vehicle_infos(db: TrafficDB, vehicle_infos: List[Dict[str, Any]]) -> None:
    for info in vehicle_infos:
        vid = info.get("id", "")
        if vid not in db.vehicles:
            continue
        vehicle = db.vehicles[vid]
        old_edge = vehicle.get("current_edge", "")
        new_edge = info.get("current_edge", "")
        vehicle["current_x"] = info.get("current_x", 0.0)
        vehicle["current_y"] = info.get("current_y", 0.0)
        vehicle["current_zone"] = info.get("current_zone", "")
        vehicle["current_edge"] = new_edge
        vehicle["current_position"] = info.get("current_position", 0.0)

        dest_edge = vehicle.get("destination_edge", "")
        full_route = vehicle.get("route", [])
        if new_edge and new_edge in full_route:
            idx = full_route.index(new_edge)
            route_left = full_route[idx + 1:]
            if new_edge == dest_edge:
                route_left = []
        else:
            route_left = info.get("route_left", [])
        vehicle["route_left"] = route_left

        route_length_left = 0.0
        dest_position = vehicle.get("destination_position", 0.0)
        curr_pos = info.get("current_position", 0.0)
        if new_edge == dest_edge:
            route_length_left = max(0.0, dest_position - curr_pos)
        elif new_edge:
            edge_len = db.road_edges.get(new_edge, {}).get("length", 0.0)
            route_length_left = max(0.0, edge_len - curr_pos)
            for e in route_left[:-1]:
                route_length_left += db.road_edges.get(e, {}).get("length", 0.0)
            if route_left and route_left[-1] == dest_edge:
                route_length_left += dest_position
        vehicle["route_length_left"] = route_length_left

        old_speed = vehicle.get("speed", 0.0)
        new_speed = info.get("speed", 0.0)
        vehicle["acceleration"] = (
            (new_speed - old_speed) / ACCELERATION_INTERVAL_SEC
            if ACCELERATION_INTERVAL_SEC > 0 else 0.0
        )
        vehicle["speed"] = new_speed

        if old_edge != new_edge:
            if old_edge:
                db.remove_vehicle_from_edge(vid, old_edge)
            if new_edge:
                db.add_vehicle_to_edge(vid, new_edge)
        elif new_edge:
            db.update_road_stats(new_edge)

        if new_edge == dest_edge and route_length_left <= 0.0:
            db.remove_vehicle(vid)


def _update_edge_demand(db: TrafficDB) -> None:
    for edge in db.road_edges.values():
        edge["edge_demand"] = 0.0
    for vehicle in db.vehicles.values():
        route_left = vehicle.get("route_left", [])
        if not route_left:
            continue
        current_edge = vehicle.get("current_edge", "")
        current_position = vehicle.get("current_position", 0.0)
        t = 0.0
        if current_edge:
            e = db.road_edges.get(current_edge, {})
            length = e.get("length", 0.0)
            speed = e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0
            t += max(0.0, length - current_position) / speed
        for edge_id in route_left:
            e = db.road_edges.get(edge_id, {})
            if not e:
                continue
            length = e.get("length", 0.0)
            speed = e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0
            e["edge_demand"] = e.get("edge_demand", 0.0) + 1.0 / (1.0 + t / EDGE_DEMAND_TAU_SEC)
            t += length / speed


def _load_segment_to_map(
    vehicle_id: str,
    seg: Dict[str, Any],
    base_ts: int,
    road_edges: Dict[str, Dict[str, Any]],
    timestamp_to_vehicles: Any,
    network_parser: Any,
    edge_shapes: Dict[str, List],
) -> Optional[Tuple[Dict[str, Any], int, Optional[int]]]:
    route_edges = seg.get("route_edges", [])
    sumo_route_gps = seg.get("sumo_route_gps", [])
    if not sumo_route_gps:
        return None

    start_ts = seg.get("starting_timestamp", base_ts)
    GPS_INTERVAL = 15
    prev_speed = 0.0

    first_pt = sumo_route_gps[0]
    last_pt = sumo_route_gps[-1]
    origin_edge = first_pt.get("edge_id", "")
    destination_edge = last_pt.get("edge_id", "")
    origin_x, origin_y = 0.0, 0.0
    destination_x, destination_y = 0.0, 0.0

    fc = first_pt.get("coordinates", [])
    if fc and len(fc) >= 2:
        oxy = network_parser.gps_to_sumo_coords(fc[0], fc[1])
        if oxy:
            origin_x, origin_y = oxy[0], oxy[1]
    origin_position = _position_on_edge_from_coords(origin_x, origin_y, origin_edge, edge_shapes) or 0.0
    if origin_edge:
        origin_position = min(max(origin_position, 0.0), road_edges.get(origin_edge, {}).get("length", 0.0))

    lc = last_pt.get("coordinates", [])
    if lc and len(lc) >= 2:
        dxy = network_parser.gps_to_sumo_coords(lc[0], lc[1])
        if dxy:
            destination_x, destination_y = dxy[0], dxy[1]
    destination_position = _position_on_edge_from_coords(
        destination_x, destination_y, destination_edge, edge_shapes
    ) or 0.0
    if destination_edge:
        destination_position = min(
            max(destination_position, 0.0),
            road_edges.get(destination_edge, {}).get("length", 0.0),
        )

    node = create_vehicle_node(vehicle_id)
    node["origin_start_sec"] = base_ts
    node["origin_edge"] = origin_edge
    node["origin_position"] = origin_position
    node["origin_x"] = origin_x
    node["origin_y"] = origin_y
    node["destination_edge"] = destination_edge
    node["destination_position"] = destination_position
    node["destination_x"] = destination_x
    node["destination_y"] = destination_y
    node["route"] = route_edges
    node["route_length"] = sum(road_edges.get(e, {}).get("length", 0.0) for e in route_edges)

    seg_first_ts = sumo_route_gps[0].get("timestamp", start_ts)
    seg_last_ts: Optional[int] = None

    for i, pt in enumerate(sumo_route_gps):
        edge_id = pt.get("edge_id", "")
        coords = pt.get("coordinates", [])
        speed = pt.get("speed", 0.0)
        ts = pt.get("timestamp", start_ts + i * GPS_INTERVAL)

        current_x, current_y = 0.0, 0.0
        if coords and len(coords) >= 2:
            sumo_xy = network_parser.gps_to_sumo_coords(coords[0], coords[1])
            if sumo_xy:
                current_x, current_y = sumo_xy[0], sumo_xy[1]

        position_on_edge = _position_on_edge_from_coords(current_x, current_y, edge_id, edge_shapes)
        edge_len = road_edges.get(edge_id, {}).get("length", 0.0)
        if position_on_edge is None:
            position_on_edge = 0.0 if edge_len <= 0 else min(0.5 * edge_len, edge_len)
        else:
            position_on_edge = min(max(position_on_edge, 0.0), edge_len)

        route_left = route_edges[route_edges.index(edge_id):] if edge_id in route_edges else route_edges
        if route_left:
            edge_lengths = [road_edges.get(e, {}).get("length", 0.0) for e in route_left]
            route_length_left = (edge_len - position_on_edge) + sum(edge_lengths[1:])
        else:
            route_length_left = 0.0

        acceleration = (speed - prev_speed) / GPS_INTERVAL if GPS_INTERVAL > 0 else 0.0
        prev_speed = speed
        current_zone = road_edges.get(edge_id, {}).get("zone", "") or ""

        vehicle_info = _build_vehicle_info(
            vehicle_id=vehicle_id,
            speed=speed,
            acceleration=acceleration,
            current_x=current_x,
            current_y=current_y,
            current_zone=current_zone,
            current_edge=edge_id,
            current_position=position_on_edge,
            route_left=route_left,
            route_length_left=route_length_left,
        )
        if ts not in timestamp_to_vehicles:
            timestamp_to_vehicles[ts] = []
        timestamp_to_vehicles[ts].append(vehicle_info)

        pt_ts = pt.get("timestamp", ts)
        if pt_ts is not None and (seg_last_ts is None or pt_ts > seg_last_ts):
            seg_last_ts = pt_ts

    return (node, seg_first_ts, seg_last_ts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_sumo_home(sumo_home_arg: Optional[Path]) -> Optional[Path]:
    if sumo_home_arg and (Path(sumo_home_arg) / "tools").exists():
        return Path(sumo_home_arg).resolve()
    from src.utils.sumo_detector import auto_detect_sumo_home
    found = auto_detect_sumo_home()
    if found and (Path(found) / "tools").exists():
        return Path(found).resolve()
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Porto parquet/CSV to step JSON files (v2 pipeline)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file: .parquet (Porto parquet) or .csv (pre-sorted by timestamp)",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("examples/porto_conversion/config/porto.net.xml"),
        help="Path to the SUMO network file (.net.xml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for step JSON files",
    )
    parser.add_argument(
        "--sampling-period",
        type=int,
        default=30,
        metavar="SEC",
        help="Snapshot interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        metavar="N",
        help="Trajectory conversion worker processes (default: CPU-1)",
    )
    parser.add_argument(
        "--writer-workers",
        type=int,
        default=min(4, multiprocessing.cpu_count()),
        metavar="N",
        help="Snapshot writer worker processes (default: min(4, CPU))",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable all parallelism (single-threaded, for debugging)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write gzip-compressed JSON (.json.gz)",
    )
    parser.add_argument(
        "--sumo-home",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to SUMO installation (SUMO_HOME)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N input rows (useful for validation runs)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    sumo_home = _resolve_sumo_home(args.sumo_home)
    if sumo_home:
        sys.path.insert(0, str(sumo_home / "tools"))

    try:
        import sumolib
        net = sumolib.net.readNet(str(args.network))
        junctions = build_junctions_map(net)
        road_edges = build_edges_map(net)
    except ImportError as e:
        print(f"sumolib not available: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load network {args.network}: {e}", file=sys.stderr)
        sys.exit(1)

    if not junctions or not road_edges:
        print("Network load produced empty junctions or edges.", file=sys.stderr)
        sys.exit(1)

    try:
        from src.utils.network_parser import NetworkParser
        network_parser = NetworkParser(str(args.network))
        conv = network_parser.conv_boundary
        bounds = network_parser.get_bounds()
        y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
        y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)
        edges_data = build_edges_data(network_parser)
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        node_positions = build_node_positions(network_parser)
    except Exception as e:
        print(f"Failed to load NetworkParser: {e}", file=sys.stderr)
        sys.exit(1)

    db = TrafficDB(junctions, road_edges)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.output / "snapshots"
    labels_dir = args.output / "labels"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    static_data = build_static_json(db)
    compress = args.compress
    if compress:
        with gzip.open(args.output / "static.json.gz", "wt", encoding="utf-8") as f:
            json.dump(static_data, f, separators=(",", ":"))
    else:
        with open(args.output / "static.json", "w", encoding="utf-8") as f:
            json.dump(static_data, f, indent=2)
    print(f"static.json written ({len(junctions)} junctions, {len(road_edges)} edges)", flush=True)

    # --- State ---
    timestamp_to_vehicles: Dict[int, List[Dict[str, Any]]] = (
        _SortedDict() if _USE_SORTED_DICT else {}
    )
    vehicle_pending: Dict[str, Tuple[Dict[str, Any], int, Optional[int]]] = {}
    vehicle_seg_last_ts: Dict[str, int] = {}
    vehicle_last_ts: Dict[str, int] = {}
    global_ts = 0
    sampling_period = args.sampling_period
    use_parallel = not args.no_parallel

    def _first_ts_key() -> Optional[int]:
        if not timestamp_to_vehicles:
            return None
        if _USE_SORTED_DICT:
            return next(iter(timestamp_to_vehicles.keys()))
        return min(timestamp_to_vehicles.keys())

    def _pop_first() -> Tuple[int, List[Dict[str, Any]]]:
        if _USE_SORTED_DICT:
            k, v = timestamp_to_vehicles.popitem(0)
            return k, v
        first_ts = min(timestamp_to_vehicles.keys())
        return first_ts, timestamp_to_vehicles.pop(first_ts)

    def _traj_first_ts(rec: Any, base_ts: int) -> Optional[int]:
        t = None
        for seg in rec.get("segments", []):
            srp = seg.get("sumo_route_gps", [])
            if srp:
                ts = srp[0].get("timestamp", base_ts)
                if t is None or ts < t:
                    t = ts
        return t

    def _load_trajectory(trip_num: int, polyline: List, ts: Optional[int], rec: Any) -> None:
        if not rec or ts is None:
            return
        trajectory_id = rec.get("trajectory_id", trip_num)
        base_ts = ts
        for seg_idx, seg in enumerate(rec.get("segments", [])):
            vehicle_id = f"veh_{trajectory_id}_{seg_idx}"
            result = _load_segment_to_map(
                vehicle_id, seg, base_ts, db.road_edges,
                timestamp_to_vehicles, network_parser, edge_shapes,
            )
            if result:
                node, seg_first_ts, seg_last_ts = result
                vehicle_pending[vehicle_id] = (node, seg_first_ts, seg_last_ts)
                if seg_last_ts is not None:
                    vehicle_seg_last_ts[vehicle_id] = seg_last_ts

    # --- Writer pool ---
    writer_pool: Optional[multiprocessing.pool.Pool] = None
    if use_parallel:
        writer_pool = multiprocessing.Pool(processes=args.writer_workers)

    def _emit_snapshot(ts_val: int, step_data: Dict[str, Any], label_data: Dict[str, Any]) -> None:
        if not step_data.get("nodes"):
            print(f"skipping empty step at ts {ts_val} (no vehicles)", flush=True)
            return
        if writer_pool is not None:
            writer_pool.apply_async(
                _write_snapshot,
                (str(snapshots_dir), str(labels_dir), ts_val, step_data, label_data, compress),
            )
        else:
            _write_snapshot(str(snapshots_dir), str(labels_dir), ts_val, step_data, label_data, compress)

    def _emit_current_state(ts_val: int) -> None:
        # Only include vehicles whose destination timestamp is known — guarantees step↔label match
        valid_vids = frozenset(db.vehicles) & frozenset(vehicle_seg_last_ts)
        step_data = build_step_json(db, ts_val, valid_vids)
        label_data = build_label_json(db, ts_val, vehicle_seg_last_ts, valid_vids)
        _emit_snapshot(ts_val, step_data, label_data)

    def _process_batch_up_to(boundary: int) -> None:
        """Pop all GPS batches with ts <= boundary and apply to DB."""
        while timestamp_to_vehicles and _first_ts_key() is not None and _first_ts_key() <= boundary:
            first_ts, vehicle_infos = _pop_first()
            for info in vehicle_infos:
                vid = info.get("id", "")
                if vid in vehicle_pending and vehicle_pending[vid][1] == first_ts:
                    _node, seg_first, seg_last = vehicle_pending[vid]
                    # Only add to DB if destination timestamp is known; otherwise step/label would mismatch
                    if seg_last is not None:
                        db.add_vehicle(vid, _node)
                        print(
                            f"vehicle {vid} added to db | route ts: {seg_first} .. {seg_last}",
                            flush=True,
                        )
                    del vehicle_pending[vid]
                if vehicle_seg_last_ts.get(vid) == first_ts:
                    vehicle_last_ts[vid] = first_ts
            update_db_from_vehicle_infos(db, vehicle_infos)

    # --- Conversion pipeline ---
    row_stream = stream_input(args.input)
    if args.limit is not None:
        row_stream = itertools.islice(row_stream, args.limit)
        print(f"--limit {args.limit}: will stop after {args.limit} input rows.", flush=True)

    if use_parallel:
        def _task_iter() -> Iterator[Tuple[int, List, Optional[int]]]:
            for row_num, polyline, ts in row_stream:
                if ts is not None:
                    yield (row_num, polyline, ts)

        conv_pool = multiprocessing.Pool(
            processes=args.workers,
            initializer=_mp_init_worker,
            initargs=(str(args.network),),
        )
        conversion_iter = conv_pool.imap(_mp_convert_one, _task_iter(), chunksize=32)
    else:
        conv_pool = None
        conversion_iter = None

    conversion_done = False
    pending_traj: Optional[Tuple[int, List, Optional[int], Any]] = None
    last_ts_seen: Optional[int] = None
    traj_processed = 0
    _prog_last = [0, 0.0]

    def _next_trajectory() -> Optional[Tuple[int, List, Optional[int], Any]]:
        nonlocal last_ts_seen, traj_processed
        if use_parallel and conversion_iter is not None:
            while True:
                try:
                    item = next(conversion_iter)
                except StopIteration:
                    return None
                if item is None:
                    continue
                trip_num, polyline, ts, rec = item
                if last_ts_seen is not None and ts < last_ts_seen:
                    print(
                        f"Error: input not sorted by timestamp at trip {trip_num} "
                        f"(ts={ts} < prev={last_ts_seen}). Use a pre-sorted file.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                last_ts_seen = ts
                traj_processed += 1
                now = time.monotonic()
                if traj_processed <= 1 or now - _prog_last[1] >= 8.0:
                    print(f"Trajectory {trip_num} converted (total: {traj_processed})", flush=True)
                    _prog_last[1] = now
                return item
        else:
            for row_num, polyline, ts in row_stream:
                if ts is None:
                    continue
                if last_ts_seen is not None and ts < last_ts_seen:
                    print(
                        f"Error: input not sorted by timestamp at row {row_num}.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                last_ts_seen = ts
                rec = convert_trajectory(
                    row_num, polyline, ts, network_parser, edges_data,
                    edge_shapes, node_positions, y_min, y_max, use_polygon=False,
                )
                if rec:
                    traj_processed += 1
                    return (row_num, polyline, ts, rec)
            return None

    def _expire_stale_vehicles(traj_cut: Optional[int]) -> None:
        """Remove DB vehicles whose last GPS event fell at or before traj_cut."""
        if traj_cut is None:
            return
        for vid in list(db.vehicles.keys()):
            last_val = vehicle_last_ts.get(vid)
            if last_val is not None and last_val <= traj_cut:
                db.remove_vehicle(vid)
                vehicle_last_ts.pop(vid, None)

    def _load_trajectories_through(boundary: int) -> None:
        """
        Pull trajectories from the conversion pool and load them into
        timestamp_to_vehicles until the next unloaded trajectory starts
        strictly after `boundary` (or conversion is exhausted).

        This guarantees that the snapshot at `boundary` sees every vehicle
        whose first GPS timestamp falls within [0, boundary], regardless of
        the order in which the conversion pool returns results.
        """
        nonlocal pending_traj, conversion_done
        while not conversion_done:
            if pending_traj is None:
                pending_traj = _next_trajectory()
                if pending_traj is None:
                    conversion_done = True
                    return
            _, _, traj_ts, rec = pending_traj
            traj_first_ts = _traj_first_ts(rec, traj_ts or 0)
            if traj_first_ts is not None and traj_first_ts > boundary:
                return  # next trajectory starts after boundary; hold it for later
            traj_cut = traj_first_ts if traj_first_ts is not None else traj_ts
            _expire_stale_vehicles(traj_cut)
            _load_trajectory(*pending_traj)
            pending_traj = None

    # Seed: load trajectories until we have at least one event in the map
    while not timestamp_to_vehicles and not conversion_done:
        if pending_traj is None:
            pending_traj = _next_trajectory()
            if pending_traj is None:
                conversion_done = True
                break
        _, _, traj_ts, rec = pending_traj
        traj_first_ts = _traj_first_ts(rec, traj_ts or 0)
        _expire_stale_vehicles(traj_first_ts if traj_first_ts is not None else traj_ts)
        _load_trajectory(*pending_traj)
        pending_traj = None

    # Main aggregation loop
    while True:
        next_ts_in_map = _first_ts_key()

        if next_ts_in_map is None:
            if conversion_done:
                break
            # Map is empty. If we already have a pending trajectory, load it
            # unconditionally to seed the map — even if it starts after the current
            # boundary (global_ts will re-align below). Without this, a pending_traj
            # whose first_ts > boundary causes an infinite spin: _load_trajectories_through
            # returns immediately every iteration because its guard fires.
            if pending_traj is not None:
                _, _, traj_ts, rec = pending_traj
                traj_first_ts = _traj_first_ts(rec, traj_ts or 0)
                _expire_stale_vehicles(traj_first_ts if traj_first_ts is not None else traj_ts)
                _load_trajectory(*pending_traj)
                pending_traj = None
                continue
            # No pending trajectory yet; fetch one from the pool (may block briefly).
            pending_traj = _next_trajectory()
            if pending_traj is None:
                conversion_done = True
            continue

        if global_ts == 0:
            global_ts = next_ts_in_map
        next_boundary = global_ts + sampling_period

        # Before emitting the snapshot at next_boundary, exhaust all trajectories
        # whose first GPS timestamp falls within the current window. Without this,
        # a trajectory whose imap result arrives later than an earlier trajectory
        # (because of parallel conversion) would be missing from the snapshot even
        # though its trip started before the snapshot boundary.
        _load_trajectories_through(next_boundary)

        next_ts_in_map = _first_ts_key()  # re-read after loading more trajectories
        if next_ts_in_map is None:
            if conversion_done:
                break
            continue

        if next_ts_in_map > next_boundary:
            # No events in [global_ts, next_boundary]. Advance without emitting.
            # Fast-forward over large gaps to avoid spinning through empty periods.
            gap_steps = max(0, (next_ts_in_map - next_boundary) // sampling_period - 1)
            global_ts = next_boundary + gap_steps * sampling_period
            continue

        _process_batch_up_to(next_boundary)
        global_ts = next_boundary
        _update_edge_demand(db)
        _emit_current_state(global_ts)

    # Drain remaining events that arrived after the last trajectory was loaded
    while timestamp_to_vehicles and _first_ts_key() is not None:
        next_boundary = global_ts + sampling_period
        _process_batch_up_to(next_boundary)
        global_ts = next_boundary
        _update_edge_demand(db)
        _emit_current_state(global_ts)

    print(f"Conversion finished: {traj_processed} trajectories processed.", flush=True)

    # Shut down pools
    if writer_pool is not None:
        writer_pool.close()
        writer_pool.join()
    if conv_pool is not None:
        conv_pool.close()
        conv_pool.join()


if __name__ == "__main__":
    main()
