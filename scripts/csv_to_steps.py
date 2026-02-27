#!/usr/bin/env python3
"""
CSV → Step (Snapshot) Files — Direct conversion script.

Converts train CSV to step JSONs without intermediate traj files.
"""

import argparse
import csv
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sortedcontainers import SortedDict as _SortedDict
    _USE_SORTED_DICT = True
except ImportError:
    _SortedDict = None
    _USE_SORTED_DICT = False

try:
    from tqdm import tqdm
except ImportError:
    class _NullPbar:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return _NullPbar()
        return iterable

from src.utils.route_finding import project_point_onto_polyline_with_segment_and_t
from src.utils.trajectory_converter import _parse_csv_row
from src.utils.traffic_db import TrafficDB


def count_csv_rows(
    csv_path: Path,
    start_traj: int,
    last_traj: Optional[int],
) -> Optional[int]:
    """
    Count data rows in range [start_traj, last_traj] without loading into memory.
    Uses csv.reader for correct handling of multi-line fields.
    Returns None only if the file is empty or missing.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            count = 0
            for i, _ in enumerate(reader, 1):
                if i < start_traj:
                    continue
                if last_traj is not None and i > last_traj:
                    break
                count += 1
        return count
    except (OSError, csv.Error):
        return None


def _stream_sorted_rows(
    sorted_csv_path: Path,
    ts_idx: Optional[int],
) -> Iterator[Tuple[int, List, Optional[int]]]:
    """
    Stream (trip_num, polyline, timestamp) one row at a time from sorted file.
    trip_num is the 1-based row index in the sorted file.
    O(1) memory - never loads full dataset.
    """
    with open(sorted_csv_path, "r", encoding="utf-8", newline="") as csv_f:
        reader = csv.reader(csv_f)
        next(reader, None)  # skip header
        for i, row in enumerate(reader, 1):
            polyline, timestamp = _parse_csv_row(row, ts_idx)
            if polyline:
                yield i, polyline, timestamp


def _stream_csv_rows(
    csv_path: Path,
    start_traj: int,
    last_traj: Optional[int],
    ts_idx: Optional[int],
) -> Iterator[Tuple[int, List, Optional[int]]]:
    """
    Stream (trip_num, polyline, timestamp) one row at a time from CSV.
    Use when data is already sorted. O(1) memory.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for i, row in enumerate(reader, 1):
            if i < start_traj:
                continue
            if last_traj is not None and i > last_traj:
                break
            polyline, timestamp = _parse_csv_row(row, ts_idx)
            if polyline:
                yield i, polyline, timestamp


def load_and_sort_csv(
    csv_path: Path,
    start_traj: int,
    last_traj: Optional[int],
    is_sorted: bool,
    save_sorted_path: Optional[Path],
) -> Iterator[Tuple[int, List, Optional[int]]]:
    """
    Sort CSV by timestamp, then return an iterator that streams one row at a time.
    When is_sorted: stream directly from source (no load).
    When not is_sorted: load, sort, write to file, then stream from file (memory freed after write).
    """
    # Load for sorting (unavoidable when not pre-sorted)
    rows_with_raw: List[Tuple[List[str], int, Optional[int]]] = []
    header = None
    ts_idx = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return iter([])
        for i, h in enumerate(header):
            if str(h).strip('"') == "TIMESTAMP":
                ts_idx = i
                break

    if is_sorted:
        return _stream_csv_rows(csv_path, start_traj, last_traj, ts_idx)

    total = count_csv_rows(csv_path, start_traj, last_traj)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        with tqdm(desc="Reading CSV", unit="rows", total=total) as pbar:
            for i, row in enumerate(reader, 1):
                pbar.update(1)
                if i > (last_traj or float("inf")):
                    break
                if i < start_traj:
                    continue
                polyline, timestamp = _parse_csv_row(row, ts_idx)
                if polyline:
                    rows_with_raw.append((row, i, timestamp))

    # Sort by timestamp, then trip_num
    with tqdm(total=1, desc="Sorting", unit="step") as pbar:
        rows_with_raw.sort(key=lambda x: (x[2] if x[2] is not None else 0, x[1]))
        pbar.update(1)

    if save_sorted_path is None:
        raise ValueError("save_sorted_path required when sorting (use default or --save-sorted)")

    save_sorted_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_sorted_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        for raw_row, _, _ in tqdm(rows_with_raw, desc="Saving sorted", unit="rows"):
            writer.writerow(raw_row)

    # rows_with_raw goes out of scope here - memory freed
    return _stream_sorted_rows(save_sorted_path, ts_idx)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert train CSV to step JSON files (direct pipeline, no traj intermediates)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Porto/dataset/train_sorted.csv"),
        help="Path to the train CSV dataset (default: Porto/dataset/train_sorted.csv)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start trajectory number (1-based, inclusive). Default: 1",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Last trajectory number (1-based, inclusive). Default: process to end of file",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("/home/guy/Projects/Traffic/Develop/Projects/PortoForSumo/config/porto.net.xml"),
        help="Path to the SUMO network file (.net.xml)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output folder for step JSON files",
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        dest="is_sorted",
        default=True,
        help="Assume dataset is already sorted by timestamp (default: True)",
    )
    parser.add_argument(
        "--no-sorted",
        action="store_false",
        dest="is_sorted",
        help="Sort the dataset by timestamp before processing (use with --save-sorted)",
    )
    parser.add_argument(
        "--save-sorted",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to save the sorted dataset. Default: same folder as CSV, with _sorted suffix. Only used when --sorted is NOT set",
    )
    parser.add_argument(
        "--sumo-home",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to SUMO installation (SUMO_HOME). If not set, tries: $SUMO_HOME, /usr/shared/sumo, /usr/share/sumo",
    )
    parser.add_argument(
        "--sampling-period",
        type=int,
        default=30,
        metavar="SEC",
        help="Step snapshot interval in seconds. Default: 30",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (cleaner output with print statements)",
    )
    return parser.parse_args()


def build_junctions_map(net: Any) -> Dict[str, Dict[str, Any]]:
    """
    Build map of all junctions from SUMO network.
    Format: {junction_id: {id, x, y, type, zone, incoming, outgoing}}
    """
    junctions: Dict[str, Dict[str, Any]] = {}
    for node in net.getNodes():
        nid = node.getID()
        if nid.startswith(":"):
            continue  # skip internal nodes
        coord = node.getCoord()
        junctions[nid] = {
            "id": nid,
            "x": coord[0],
            "y": coord[1],
            "type": node.getType() or "priority",
            "zone": "",  # Porto/OSM networks may not have zone; derive from coords if needed
            "incoming": [e.getID() for e in node.getIncoming()],
            "outgoing": [e.getID() for e in node.getOutgoing()],
        }
    return junctions


def create_vehicle_node(vehicle_id: str) -> Dict[str, Any]:
    """
    Create a vehicle node dict with vehicle_type=passenger, all other values zeroed.
    """
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


def _get_edge_shape(edge_id: str, edge_shapes: Dict[str, List]) -> Optional[List]:
    """Get shape for edge (base or lane-level). Returns list of [x,y] points."""
    if edge_id in edge_shapes:
        return edge_shapes[edge_id]
    for eid, shape in edge_shapes.items():
        if "#" in eid and eid.split("#")[0] == edge_id:
            return shape
    if "#" not in edge_id and f"{edge_id}#0" in edge_shapes:
        return edge_shapes[f"{edge_id}#0"]
    return None


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
    """
    Build VehicleInfo dict for timestamp_to_vehicles map.
    Dynamic fields only: id, speed, acceleration, current_x, current_y, current_zone,
    current_edge, current_position, route_left, route_length_left.
    """
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


def _position_on_edge_from_coords(
    x: float, y: float, edge_id: str, edge_shapes: Dict[str, List]
) -> Optional[float]:
    """
    Project vehicle coords onto edge shape and return distance from edge start (0 to edge_length).
    """
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


def _strip_lane_suffix(edge_id: str) -> str:
    """Strip #lane_index from edge ID. E.g. '-1000487472#0' -> '-1000487472'."""
    if "#" in edge_id:
        return edge_id.split("#")[0]
    return edge_id


def build_edges_map(net: Any) -> Dict[str, Dict[str, Any]]:
    """
    Build map of all edges (roads) from SUMO network.
    Lanes of the same edge are aggregated: id has no #, num_lanes is total lane count.
    Format: {base_edge_id: {id, from, to, speed, length, num_lanes, zone, density, avg_speed, vehicles_on_road}}
    """
    # Group lane-level edges by base ID
    by_base: Dict[str, List[Any]] = {}
    for edge in net.getEdges():
        eid = edge.getID()
        if edge.getFunction() == "internal":
            continue  # skip internal junction edges
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
            "edge_type": 0,  # 0 = static road edge (junction to junction)
            "speed": first.getSpeed(),
            "length": first.getLength(),
            "num_lanes": sum(e.getLaneNumber() for e in lane_edges),
            "zone": "",
            "density": 0.0,
            "avg_speed": 0.0,
            "edge_demand": 0.0,  # TBD: score for how/when road is in active vehicles' routes
            "vehicles_on_road": [],
        }
    return edges


def create_dynamic_edges(
    edges: Dict[str, Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create dynamic edges for roads with vehicles.
    For each edge where vehicles_on_road is non-empty:
    - Sort vehicles by current_position on the edge (ascending = from junction to to junction)
    - Build chain: junction_from -> vehicle_0 -> vehicle_1 -> ... -> junction_to
    - edge_type 1: junction -> first vehicle
    - edge_type 2: vehicle -> vehicle (by position)
    - edge_type 3: last vehicle -> junction

    Returns list of dynamic edge dicts with: id, from, to, edge_type, ...
    """
    dynamic: List[Dict[str, Any]] = []
    dyn_idx = 0

    for edge_id, edge_data in edges.items():
        veh_ids = edge_data.get("vehicles_on_road", [])
        if not veh_ids:
            continue

        junction_from = edge_data.get("from", "")
        junction_to = edge_data.get("to", "")

        # Sort vehicles by position on edge (lower = closer to from junction)
        sorted_vehicles = sorted(
            (vid for vid in veh_ids if vid in vehicles),
            key=lambda vid: vehicles[vid].get("current_position", 0.0),
        )
        if not sorted_vehicles:
            continue

        # Chain: junction_from -> v0 -> v1 -> ... -> junction_to
        prev = junction_from
        for i, vid in enumerate(sorted_vehicles):
            if i == 0:
                edge_type = 1  # junction -> vehicle
            else:
                edge_type = 2  # vehicle -> vehicle

            dyn_id = f"dyn_{dyn_idx}"
            dyn_idx += 1
            dynamic.append({
                "id": dyn_id,
                "from": prev,
                "to": vid,
                "edge_type": edge_type,
                "speed": 0.0,
                "length": 0.0,
                "num_lanes": 0,
                "zone": "",
                "density": 0.0,
                "avg_speed": 0.0,
                "vehicles_on_road": [],
            })
            prev = vid

        # Last: last vehicle -> junction_to
        dynamic.append({
            "id": f"dyn_{dyn_idx}",
            "from": prev,
            "to": junction_to,
            "edge_type": 3,  # vehicle -> junction
            "speed": 0.0,
            "length": 0.0,
            "num_lanes": 0,
            "zone": "",
            "density": 0.0,
            "avg_speed": 0.0,
            "vehicles_on_road": [],
        })
        dyn_idx += 1

    return dynamic


ACCELERATION_INTERVAL_SEC = 15.0
EDGE_DEMAND_TAU_SEC = 600.0  # time scale for edge_demand decay: 1/(1 + t/τ)


def update_db_from_vehicle_infos(db: TrafficDB, vehicle_infos: List[Dict[str, Any]]) -> None:
    """
    Update DB with dynamic vehicle info from a timestamp batch.
    1. Update current_* with new values.
    2. Update route_left: remove all edges before current edge inclusively.
    3. Update route_length_left: from current_position on current edge to destination_position on destination edge.
    4. Calculate acceleration from existing and new speed (15 second interval).
    5. Update speed.
    6. If vehicle changed edges: remove from old edge, add to new edge; density and avg_speed updated via _update_edge_stats.
    """
    for info in vehicle_infos:
        vid = info.get("id", "")
        if vid not in db.vehicles:
            continue
        vehicle = db.vehicles[vid]

        old_edge = vehicle.get("current_edge", "")
        new_edge = info.get("current_edge", "")

        # 1. Update current_*
        vehicle["current_x"] = info.get("current_x", 0.0)
        vehicle["current_y"] = info.get("current_y", 0.0)
        vehicle["current_zone"] = info.get("current_zone", "")
        vehicle["current_edge"] = new_edge
        vehicle["current_position"] = info.get("current_position", 0.0)

        # 2. Update route_left: edges after current edge (exclusive)
        current_edge = info.get("current_edge", "")
        full_route = vehicle.get("route", [])
        if current_edge and current_edge in full_route:
            idx = full_route.index(current_edge)
            route_left = full_route[idx + 1:]
        else:
            route_left = info.get("route_left", [])
        vehicle["route_left"] = route_left

        # 3. Update route_length_left: from current_position to destination_position along route
        route_length_left = 0.0
        dest_edge = vehicle.get("destination_edge", "")
        dest_position = vehicle.get("destination_position", 0.0)
        curr_pos = info.get("current_position", 0.0)
        if current_edge == dest_edge:
            route_length_left = max(0.0, dest_position - curr_pos)
        elif current_edge:
            edge_len = db.road_edges.get(current_edge, {}).get("length", 0.0)
            route_length_left = max(0.0, edge_len - curr_pos)
            for e in route_left[:-1]:
                route_length_left += db.road_edges.get(e, {}).get("length", 0.0)
            if route_left and route_left[-1] == dest_edge:
                route_length_left += dest_position
        vehicle["route_length_left"] = route_length_left

        # 4 & 5. Acceleration from existing and new speed, then update speed
        old_speed = vehicle.get("speed", 0.0)
        new_speed = info.get("speed", 0.0)
        vehicle["acceleration"] = (
            (new_speed - old_speed) / ACCELERATION_INTERVAL_SEC
            if ACCELERATION_INTERVAL_SEC > 0
            else 0.0
        )
        vehicle["speed"] = new_speed

        # 6. Edge change last (vehicle has correct speed) → no separate recompute needed
        if old_edge != new_edge:
            if old_edge:
                db.remove_vehicle_from_edge(vid, old_edge)
            if new_edge:
                db.add_vehicle_to_edge(vid, new_edge)
        elif new_edge:
            db.update_road_stats(new_edge)  # same edge, speed changed


def _update_edge_demand(db: TrafficDB) -> None:
    """
    Update edge_demand for all road edges.
    Score = sum over vehicles of 1/(1 + t/τ) where t = time (seconds) to reach the edge, τ = EDGE_DEMAND_TAU_SEC.
    Only edges in vehicle's route_left contribute (no route_left → no demand).
    Time uses edge avg_speed if > 0 else edge speed; length/speed per edge.
    """
    for edge in db.road_edges.values():
        edge["edge_demand"] = 0.0

    for vehicle in db.vehicles.values():
        route_left = vehicle.get("route_left", [])
        if not route_left:
            continue

        current_edge = vehicle.get("current_edge", "")
        current_position = vehicle.get("current_position", 0.0)
        t = 0.0

        # Time to exit current edge
        if current_edge and len(route_left) > 0:
            e = db.road_edges.get(current_edge, {})
            length = e.get("length", 0.0)
            speed = e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0
            remaining = max(0.0, length - current_position)
            t += remaining / speed

        # For each edge in route_left: add contribution, then accumulate time
        for edge_id in route_left:
            e = db.road_edges.get(edge_id, {})
            if not e:
                continue
            length = e.get("length", 0.0)
            speed = e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0

            contribution = 1.0 / (1.0 + t / EDGE_DEMAND_TAU_SEC)
            e["edge_demand"] = e.get("edge_demand", 0.0) + contribution

            t += length / speed


def _calculate_eta_and_labels(db: TrafficDB, snapshot_timestamp: int) -> None:
    """
    Calculate and add ETA label and other labels for vehicles.
    TBD: User will define labels and computation.
    """
    # Placeholder
    pass


STATIC_EDGE_EXCLUDE = {"vehicles_on_road", "edge_demand", "avg_speed", "density"}


def build_static_json(db: TrafficDB) -> Dict[str, Any]:
    """
    Build static JSON: junctions (all features) + road edges (static features only).
    Excludes from edges: vehicles_on_road, edge_demand, avg_speed, density.
    """
    junctions = list(db.junctions.values())
    road_edges_static = [
        {k: v for k, v in e.items() if k not in STATIC_EDGE_EXCLUDE}
        for e in db.road_edges.values()
    ]
    return {
        "junctions": junctions,
        "road_edges": road_edges_static,
    }


def build_step_json(
    db: TrafficDB,
    snapshot_timestamp: int,
) -> Dict[str, Any]:
    """
    Build step snapshot JSON: vehicles only, road edge dynamic fields, dynamic edges.
    Static data (junctions, road edge static features) is in static.json.
    """
    dynamic_edges = create_dynamic_edges(db.road_edges, db.vehicles)
    road_edges_dynamic = [
        {
            "id": e["id"],
            "vehicles_on_road": e.get("vehicles_on_road", []),
            "edge_demand": e.get("edge_demand", 0.0),
            "avg_speed": e.get("avg_speed", 0.0),
            "density": e.get("density", 0.0),
        }
        for e in db.road_edges.values()
        if e.get("vehicles_on_road")
        or e.get("edge_demand", 0.0) != 0.0
        or e.get("avg_speed", 0.0) != 0.0
        or e.get("density", 0.0) != 0.0
    ]
    return {
        "step": snapshot_timestamp,
        "nodes": list(db.vehicles.values()),
        "road_edges_dynamic": road_edges_dynamic,
        "dynamic_edges": dynamic_edges,
    }


def _resolve_sumo_home(sumo_home_arg: Optional[Path]) -> Optional[Path]:
    """Resolve SUMO_HOME: arg > env > /usr/shared/sumo > /usr/share/sumo."""
    for candidate in (
        sumo_home_arg,
        (Path(os.environ["SUMO_HOME"]) if os.environ.get("SUMO_HOME") else None),
        Path("/usr/shared/sumo"),
        Path("/usr/share/sumo"),
    ):
        if candidate and (Path(candidate) / "tools").exists():
            return Path(candidate)
    return None


def _load_segment_to_map(
    vehicle_id: str,
    seg: Dict[str, Any],
    base_ts: int,
    road_edges: Dict[str, Dict[str, Any]],
    timestamp_to_vehicles: Any,  # SortedDict[int, List[Dict]]
    network_parser: Any,
    edge_shapes: Dict[str, List],
) -> Optional[Tuple[Dict[str, Any], int, Optional[int]]]:
    """
    Load segment data into timestamp_to_vehicles only. Do NOT add vehicle to DB.
    Returns (static_node, segment_first_ts, segment_last_ts) for deferred DB add, or None.
    """
    route_edges = seg.get("route_edges", [])
    sumo_route_gps = seg.get("sumo_route_gps", [])
    if not sumo_route_gps:
        return None

    start_ts = seg.get("starting_timestamp", base_ts)
    prev_speed = 0.0
    GPS_INTERVAL = 15

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
    origin_position = _position_on_edge_from_coords(
        origin_x, origin_y, origin_edge, edge_shapes
    ) or 0.0
    if origin_edge:
        L = road_edges.get(origin_edge, {}).get("length", 0.0)
        origin_position = min(max(origin_position, 0.0), L)
    lc = last_pt.get("coordinates", [])
    if lc and len(lc) >= 2:
        dxy = network_parser.gps_to_sumo_coords(lc[0], lc[1])
        if dxy:
            destination_x, destination_y = dxy[0], dxy[1]
    destination_position = _position_on_edge_from_coords(
        destination_x, destination_y, destination_edge, edge_shapes
    ) or 0.0
    if destination_edge:
        L = road_edges.get(destination_edge, {}).get("length", 0.0)
        destination_position = min(max(destination_position, 0.0), L)

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

    seg_first_ts = sumo_route_gps[0].get("timestamp", start_ts) if sumo_route_gps else start_ts
    seg_last_ts = None

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

        position_on_edge = _position_on_edge_from_coords(
            current_x, current_y, edge_id, edge_shapes
        )
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


def main():
    args = parse_args()

    # Add SUMO tools to path for sumolib (before any sumolib import)
    sumo_home = _resolve_sumo_home(args.sumo_home)
    if sumo_home:
        sys.path.insert(0, str(sumo_home / "tools"))

    net = None
    junctions = {}
    road_edges = {}

    try:
        import sumolib
        net = sumolib.net.readNet(str(args.network))
        junctions = build_junctions_map(net)
        road_edges = build_edges_map(net)
    except ImportError:
        pass
    except Exception:
        pass

    if not junctions or not road_edges:
        sys.exit(1)

    db = TrafficDB(junctions, road_edges)

    # Load NetworkParser for trajectory conversion
    network_parser = None
    edges_data = []
    edge_shapes = {}
    node_positions = {}
    y_min, y_max = 0.0, 0.0
    try:
        from src.utils.network_parser import NetworkParser
        from src.utils.route_finding import build_edges_data, build_node_positions
        network_parser = NetworkParser(str(args.network))
        conv = network_parser.conv_boundary
        bounds = network_parser.get_bounds()
        y_min = conv["y_min"] if conv else (bounds["y_min"] if bounds else 0.0)
        y_max = conv["y_max"] if conv else (bounds["y_max"] if bounds else 0.0)
        edges_data = build_edges_data(network_parser)
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        node_positions = build_node_positions(network_parser)
    except Exception:
        pass

    save_sorted_path = args.save_sorted
    if not args.is_sorted and save_sorted_path is None:
        save_sorted_path = args.csv.parent / f"{args.csv.stem}_sorted{args.csv.suffix}"

    sorted_rows = load_and_sort_csv(
        csv_path=args.csv,
        start_traj=args.start,
        last_traj=args.last,
        is_sorted=args.is_sorted,
        save_sorted_path=save_sorted_path,
    )

    sampling_period = args.sampling_period
    global_ts = 0
    vehicle_last_ts: Dict[str, int] = {}
    vehicle_pending: Dict[str, Tuple[Dict[str, Any], int, Optional[int]]] = {}
    vehicle_seg_last_ts: Dict[str, int] = {}
    timestamp_to_vehicles: Dict[int, List[Dict[str, Any]]] = (
        _SortedDict() if _USE_SORTED_DICT else {}
    )
    args.output.mkdir(parents=True, exist_ok=True)

    # Save static JSON once (junctions + road edges with static features only)
    static_data = build_static_json(db)
    static_path = args.output / "static.json.gz"
    with gzip.open(static_path, "wt", encoding="utf-8") as f:
        json.dump(static_data, f, separators=(",", ":"))

    from src.utils.trajectory_converter import convert_trajectory

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

    def _first_ts_key() -> Optional[int]:
        if not timestamp_to_vehicles:
            return None
        if _USE_SORTED_DICT:
            return next(iter(timestamp_to_vehicles.keys()))
        return min(timestamp_to_vehicles.keys())

    def _pop_first() -> Tuple[int, List[Dict[str, Any]]]:
        if _USE_SORTED_DICT:
            k, v = timestamp_to_vehicles.popitem(last=False)
            return k, v
        first_ts = min(timestamp_to_vehicles.keys())
        return first_ts, timestamp_to_vehicles.pop(first_ts)

    trajectory_iter = iter(
        tqdm(sorted_rows, desc="Processing", unit="traj", disable=args.no_progress)
    )
    pending_traj: Optional[Tuple[int, List, Optional[int], Any]] = None
    last_ts: Optional[int] = None

    while True:
        next_ts_in_map = _first_ts_key()

        if pending_traj is None:
            try:
                trip_num, polyline, ts = next(trajectory_iter)
                if ts is None:
                    sys.exit(1)
                if last_ts is not None and ts < last_ts:
                    print(
                        f"Error: CSV is not sorted by timestamp. Trajectory {trip_num} has ts {ts} < previous ts {last_ts}. "
                        "Use --sorted False to sort by timestamp, or ensure the CSV is pre-sorted by timestamp.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                last_ts = ts
                if network_parser and edges_data:
                    rec = convert_trajectory(
                        trip_num, polyline, ts, network_parser,
                        edges_data, edge_shapes, node_positions, y_min, y_max, use_polygon=False,
                    )
                    if rec:
                        pending_traj = (trip_num, polyline, ts, rec)
            except StopIteration:
                pass

        if pending_traj is not None:
            _, _, traj_ts, rec = pending_traj
            traj_first_ts = _traj_first_ts(rec, traj_ts or 0)
            load_now = next_ts_in_map is None or (
                traj_first_ts is not None and traj_first_ts < next_ts_in_map
            )
            if load_now:
                for vid in list(db.vehicles.keys()):
                    if vehicle_last_ts.get(vid, 0) < traj_ts:
                        db.remove_vehicle(vid)
                        vehicle_last_ts.pop(vid, None)
                _load_trajectory(*pending_traj)
                pending_traj = None
                continue

        if next_ts_in_map is None:
            if pending_traj is not None:
                _load_trajectory(*pending_traj)
                pending_traj = None
                continue
            break

        if global_ts == 0:
            global_ts = next_ts_in_map
        next_json_boundary = global_ts + sampling_period

        if next_ts_in_map <= next_json_boundary:
            while timestamp_to_vehicles and _first_ts_key() is not None and _first_ts_key() <= next_json_boundary:
                first_ts, vehicle_infos = _pop_first()
                for info in vehicle_infos:
                    vid = info.get("id", "")
                    if vid in vehicle_pending and vehicle_pending[vid][1] == first_ts:
                        db.add_vehicle(vid, vehicle_pending[vid][0])
                        del vehicle_pending[vid]
                        print(f"vehicle id : {vid} added to db with ts {first_ts}", flush=True)
                    if vehicle_seg_last_ts.get(vid) == first_ts:
                        vehicle_last_ts[vid] = first_ts
                ids = [v.get("id", "") for v in vehicle_infos]
                print(f"poped ts {first_ts} with vehicles: {ids}", flush=True)
                update_db_from_vehicle_infos(db, vehicle_infos)
            global_ts = next_json_boundary
            _update_edge_demand(db)
            _calculate_eta_and_labels(db, global_ts)
            step_data = build_step_json(db, global_ts)
            out_path = args.output / f"step_{global_ts:012d}.json.gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(step_data, f, separators=(",", ":"))
            print(f"creating json {out_path.name} at ts {global_ts}", flush=True)
            continue

        first_ts, vehicle_infos = _pop_first()
        for info in vehicle_infos:
            vid = info.get("id", "")
            if vid in vehicle_pending and vehicle_pending[vid][1] == first_ts:
                db.add_vehicle(vid, vehicle_pending[vid][0])
                del vehicle_pending[vid]
                print(f"vehicle id : {vid} added to db with ts {first_ts}", flush=True)
            if vehicle_seg_last_ts.get(vid) == first_ts:
                vehicle_last_ts[vid] = first_ts
        ids = [v.get("id", "") for v in vehicle_infos]
        print(f"poped ts {first_ts} with vehicles: {ids}", flush=True)
        update_db_from_vehicle_infos(db, vehicle_infos)

    while timestamp_to_vehicles and _first_ts_key() is not None:
        next_json_boundary = global_ts + sampling_period
        while timestamp_to_vehicles and _first_ts_key() is not None and _first_ts_key() <= next_json_boundary:
            first_ts, vehicle_infos = _pop_first()
            for info in vehicle_infos:
                vid = info.get("id", "")
                if vid in vehicle_pending and vehicle_pending[vid][1] == first_ts:
                    db.add_vehicle(vid, vehicle_pending[vid][0])
                    del vehicle_pending[vid]
                    print(f"vehicle id : {vid} added to db with ts {first_ts}", flush=True)
                if vehicle_seg_last_ts.get(vid) == first_ts:
                    vehicle_last_ts[vid] = first_ts
            ids = [v.get("id", "") for v in vehicle_infos]
            print(f"poped ts {first_ts} with vehicles: {ids}", flush=True)
            update_db_from_vehicle_infos(db, vehicle_infos)
        global_ts = next_json_boundary
        _update_edge_demand(db)
        _calculate_eta_and_labels(db, global_ts)
        step_data = build_step_json(db, global_ts)
        out_path = args.output / f"step_{global_ts:012d}.json.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(step_data, f, separators=(",", ":"))
        print(f"creating json {out_path.name} at ts {global_ts}", flush=True)


if __name__ == "__main__":
    main()
