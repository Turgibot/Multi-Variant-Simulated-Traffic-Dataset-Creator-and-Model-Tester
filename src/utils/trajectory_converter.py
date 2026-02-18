"""
Shared trajectory-to-JSON conversion logic.
Used by convert_trajectories_fast.py script and DatasetConversionPage GUI.
"""

import ast
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.network_parser import NetworkParser
from src.utils.route_finding import (
    EdgeSpatialIndex,
    build_edges_data,
    build_node_positions,
    compute_green_orange_edges,
    project_point_onto_polyline,
    project_point_onto_polyline_with_segment_and_t,
    shortest_path_dijkstra,
)
from src.utils.trip_validator import validate_trip_segments

GPS_INTERVAL_SEC = 15
MAX_SUMO_ROUTE_DISTANCE_M = 150.0


def _strip_lane_suffix(edge_id: str) -> str:
    """Remove #<number> lane postfix from edge ID."""
    if "#" in edge_id:
        return edge_id.split("#")[0]
    return edge_id


def _deduplicate_consecutive(items: List[str]) -> List[str]:
    """Remove consecutive duplicates: [a, a, b, b, a] -> [a, b, a]."""
    if not items:
        return []
    result = [items[0]]
    for x in items[1:]:
        if x != result[-1]:
            result.append(x)
    return result


def apply_gps_offset(lon: float, lat: float, offset_x_m: float, offset_y_m: float) -> Tuple[float, float]:
    """Apply meter offset to GPS coords. X=east-west, Y=north-south.
    Returns (adjusted_lon, adjusted_lat). Used for trajectory display and route finding.
    """
    if offset_x_m == 0 and offset_y_m == 0:
        return lon, lat
    # Meters per degree: lat ~110540m, lon ~111320*cos(lat)m
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat)) if lat != 90 else 0.0
    if m_per_deg_lon < 1e-6:
        m_per_deg_lon = 111320.0
    delta_lon = offset_x_m / m_per_deg_lon
    delta_lat = offset_y_m / m_per_deg_lat
    return lon + delta_lon, lat + delta_lat


def iter_trajectories_from_csv(
    csv_path: str, start_traj: int, last_traj: int
):
    """
    Single-pass iterator over trajectories in range [start_traj, last_traj].
    Yields (trip_num, polyline, timestamp) - no re-reading from file start.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        ts_idx = None
        for i, h in enumerate(header):
            if str(h).strip('"') == "TIMESTAMP":
                ts_idx = i
                break
        for i, row in enumerate(reader, 1):
            if i > last_traj:
                break
            if i < start_traj:
                continue
            polyline, timestamp = _parse_csv_row(row, ts_idx)
            if polyline:
                yield i, polyline, timestamp


def _parse_csv_row(row: List[str], ts_idx: Optional[int] = None) -> Tuple[Optional[List[List[float]]], Optional[int]]:
    """Parse a CSV row into (polyline, timestamp). ts_idx from header."""
    timestamp = None
    if ts_idx is not None and ts_idx < len(row):
        try:
            timestamp = int(str(row[ts_idx]).strip('"'))
        except (ValueError, TypeError):
            pass
    polyline_str = None
    for cell in row:
        s = str(cell).strip()
        if s.startswith("[[") and s.endswith("]]"):
            polyline_str = s
            break
    if not polyline_str:
        return None, None
    try:
        polyline = ast.literal_eval(polyline_str)
        if isinstance(polyline, list) and len(polyline) >= 2:
            return polyline, timestamp
    except Exception:
        pass
    return None, None


def _detect_real_start_and_end(polyline: List[List[float]]) -> Tuple[int, int]:
    if not polyline or len(polyline) < 3:
        return 0, len(polyline) - 1 if polyline else 0
    STATIC_THRESHOLD = 15.0
    R = 6371000

    def haversine_m(c1: List[float], c2: List[float]) -> float:
        lat1, lon1 = c1[0], c1[1]
        lat2, lon2 = c2[0], c2[1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    real_start = 0
    for i in range(len(polyline) - 1):
        if haversine_m(polyline[i], polyline[i + 1]) > STATIC_THRESHOLD:
            real_start = i
            break
    real_end = len(polyline) - 1
    for i in range(len(polyline) - 1, 0, -1):
        if haversine_m(polyline[i - 1], polyline[i]) > STATIC_THRESHOLD:
            real_end = i
            break
    return real_start, real_end


def _split_at_invalid_segments(polyline: List[List[float]]) -> List[List[List[float]]]:
    if not polyline or len(polyline) < 2:
        return [polyline] if polyline else []
    validation = validate_trip_segments(polyline)
    invalid_indices = set(validation.invalid_segment_indices)
    if not invalid_indices:
        return [polyline]
    segments: List[List[List[float]]] = []
    current = [polyline[0]]
    for i in range(1, len(polyline)):
        if i - 1 in invalid_indices:
            segments.append(current)
            current = [polyline[i]]
        else:
            current.append(polyline[i])
    if current:
        segments.append(current)
    return segments


def convert_trajectory(
    trip_num: int,
    polyline: List[List[float]],
    base_timestamp: Optional[int],
    network_parser: NetworkParser,
    edges_data: List[Tuple[str, Dict, List]],
    edge_shapes: Dict[str, List],
    node_positions: Dict[str, Tuple[float, float]],
    y_min: float,
    y_max: float,
    use_polygon: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    spatial_index: Optional[Any] = None,
    cancelled_callback: Optional[Any] = None,
) -> Optional[Dict]:
    """
    Convert a single trajectory to JSON record. Returns None if no valid route.

    offset_x, offset_y: GPS offset in meters (X=east, Y=north). Applied to trajectory
    GPS coords before conversion. Map and network stay fixed; only trajectory moves.
    """
    def flip_y(y: float) -> float:
        return y_max + y_min - y

    validation = validate_trip_segments(polyline)
    real_start, real_end = _detect_real_start_and_end(polyline)

    if validation.invalid_segment_count > 0:
        split_segments = _split_at_invalid_segments(polyline)
        segments = []
        seg_offsets = []
        offset = 0
        for seg in split_segments:
            if len(seg) >= 2:
                s0, s1 = _detect_real_start_and_end(seg)
                trim_seg = seg[s0 : s1 + 1]
                if len(trim_seg) >= 3:
                    segments.append(trim_seg)
                    seg_offsets.append(offset + s0)
            offset += len(seg)
    else:
        trim_poly = polyline[real_start : real_end + 1]
        if len(trim_poly) >= 3:
            segments = [trim_poly]
            seg_offsets = [real_start]
        else:
            segments = []
            seg_offsets = []

    base_ts = base_timestamp if base_timestamp is not None else 0
    trajectory_segments: List[Dict] = []

    base_edges: Optional[Set[str]] = None
    if use_polygon:
        base_edges = compute_edges_in_polygon(network_parser, segments)

    for segment, orig_offset in zip(segments, seg_offsets):
        if cancelled_callback and cancelled_callback():
            return None
        sumo_points_flipped = []
        for lon, lat in segment:
            # Apply GPS offset to trajectory coords (map/network stay fixed)
            adj_lon, adj_lat = apply_gps_offset(lon, lat, offset_x, offset_y)
            coords = network_parser.gps_to_sumo_coords(adj_lon, adj_lat)
            if coords:
                x, y = coords
                sumo_points_flipped.append((x, flip_y(y)))
        if len(sumo_points_flipped) < 2:
            continue

        orange_ids, green_ids, start_id, end_id, candidates = compute_green_orange_edges(
            edges_data,
            sumo_points_flipped,
            y_min,
            y_max,
            top_per_segment=5,
            spatial_index=spatial_index,
            filter_radius=400.0,
        )
        if not start_id or not end_id:
            continue

        goal_xy = sumo_points_flipped[-1]
        max_tries = min(5, len(candidates or []))
        path_edges = None
        path_points_flat = None
        path_point_to_edge = None

        for try_idx in range(max_tries):
            if cancelled_callback and cancelled_callback():
                return None
            cand = (candidates or [])[try_idx]
            edges_allowed = (base_edges | {cand, end_id}) if base_edges is not None else None
            candidate_path = shortest_path_dijkstra(
                network_parser,
                cand,
                end_id,
                orange_ids=orange_ids,
                green_ids=green_ids,
                node_positions=node_positions,
                goal_xy=goal_xy,
                edges_in_polygon=edges_allowed,
            )
            if not candidate_path:
                continue

            path_points = []
            point_to_edge = []
            for eid in candidate_path:
                sp = edge_shapes.get(eid)
                if sp:
                    for x_s, y_s in sp:
                        path_points.append((x_s, flip_y(y_s)))
                        point_to_edge.append(eid)
            path_points_flat_cand = [[p[0], p[1]] for p in path_points]

            all_within = True
            for px, py in sumo_points_flipped:
                proj_x, proj_y = project_point_onto_polyline(px, py, path_points_flat_cand)
                dist_m = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                if dist_m > MAX_SUMO_ROUTE_DISTANCE_M:
                    all_within = False
                    break
            if all_within:
                path_edges = candidate_path
                path_points_flat = path_points_flat_cand
                path_point_to_edge = point_to_edge
                break

        if path_edges is None or path_points_flat is None or path_point_to_edge is None:
            continue

        # Segment lengths along path (meters)
        segment_lengths: List[float] = []
        for i in range(len(path_points_flat) - 1):
            x1, y1 = path_points_flat[i][0], path_points_flat[i][1]
            x2, y2 = path_points_flat[i + 1][0], path_points_flat[i + 1][1]
            segment_lengths.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        starting_timestamp = base_ts + orig_offset * GPS_INTERVAL_SEC
        sumo_route_points: List[Dict[str, Any]] = []
        prev_cumulative = 0.0

        for i, (px, py) in enumerate(sumo_points_flipped):
            (proj_x, proj_y), seg_idx, t = project_point_onto_polyline_with_segment_and_t(
                px, py, path_points_flat
            )
            cum_before = sum(segment_lengths[:seg_idx]) if seg_idx > 0 else 0.0
            seg_len = segment_lengths[seg_idx] if seg_idx < len(segment_lengths) else 0.0
            cumulative_dist = cum_before + t * seg_len

            distance_from_previous = cumulative_dist - prev_cumulative if i > 0 else 0.0
            speed = distance_from_previous / GPS_INTERVAL_SEC if GPS_INTERVAL_SEC > 0 else 0.0
            prev_cumulative = cumulative_dist

            lon_lat = network_parser.sumo_to_gps_coords(proj_x, flip_y(proj_y))
            if lon_lat:
                raw_edge = path_point_to_edge[seg_idx] if seg_idx < len(path_point_to_edge) else ""
                edge_id = _strip_lane_suffix(raw_edge)
                sumo_route_points.append({
                    "timestamp": starting_timestamp + i * GPS_INTERVAL_SEC,
                    "edge_id": edge_id,
                    "coordinates": list(lon_lat),
                    "distance_from_previous": round(distance_from_previous, 4),
                    "speed": round(speed, 4),
                })

        route_edges_stripped = _deduplicate_consecutive(
            [_strip_lane_suffix(e) for e in path_edges]
        )
        duration_seconds = (len(segment) - 1) * GPS_INTERVAL_SEC if len(segment) > 1 else 0
        trajectory_segments.append({
            "starting_timestamp": starting_timestamp,
            "duration_seconds": duration_seconds,
            "number_of_gps_points": len(segment),
            "gps_points": segment,
            "number_of_edges": len(route_edges_stripped),
            "route_edges": route_edges_stripped,
            "number_of_sumo_route_gps_points": len(sumo_route_points),
            "sumo_route_gps": sumo_route_points,
        })

    if not trajectory_segments:
        return None

    return {
        "trajectory_id": trip_num,
        "segments": trajectory_segments,
    }


def compute_edges_in_polygon(
    network_parser: NetworkParser,
    segments: List[List[List[float]]],
) -> Optional[Set[str]]:
    """Edges inside trajectory polygon (200m padding). Returns None if polygon cannot be built."""
    if not network_parser or not segments:
        return None
    all_gps = []
    for seg in segments:
        if seg and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 2:
            for pt in seg:
                all_gps.append((pt[0], pt[1]))
    if len(all_gps) < 2:
        return None
    sumo_points = []
    for lon, lat in all_gps:
        coords = network_parser.gps_to_sumo_coords(lon, lat)
        if coords:
            sumo_points.append(coords)
    if len(sumo_points) < 2:
        return None
    min_box = _find_minimum_bounding_box_route(sumo_points, padding_meters=200.0)
    if not min_box:
        return None
    center_x, center_y, width, height, angle = min_box
    half_w, half_h = width / 2, height / 2
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotated_corners = [
        (center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a)
        for dx, dy in corners
    ]
    outside = [i for i, (x, y) in enumerate(sumo_points) if not _point_in_polygon(x, y, rotated_corners + [rotated_corners[0]])]
    if outside:
        center_x, center_y, width, height, angle = _expand_box_to_include_all_points(
            sumo_points, center_x, center_y, width, height, angle, safety_margin=100.0
        )
        half_w, half_h = width / 2, height / 2
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        rotated_corners = [
            (center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a)
            for dx, dy in corners
        ]
    polygon = rotated_corners + [rotated_corners[0]]
    edges = network_parser.get_edges()
    edge_ids: Set[str] = set()
    for edge_id, edge_data in edges.items():
        if not edge_data.get("lanes"):
            continue
        shape_points = edge_data["lanes"][0].get("shape", [])
        if len(shape_points) < 2:
            continue
        edge_in_box = False
        for pt in shape_points:
            x, y = pt[0], pt[1]
            if _point_in_polygon(x, y, polygon):
                edge_in_box = True
                break
        if not edge_in_box:
            for i in range(len(shape_points) - 1):
                x1, y1 = shape_points[i][0], shape_points[i][1]
                x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
                if _line_intersects_polygon(x1, y1, x2, y2, polygon):
                    edge_in_box = True
                    break
        if edge_in_box:
            edge_ids.add(edge_id)
    return edge_ids


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _line_segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> bool:
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _line_intersects_polygon(
    x1: float, y1: float, x2: float, y2: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    pts = polygon if polygon[0] == polygon[-1] else polygon + [polygon[0]]
    for i in range(len(pts) - 1):
        x3, y3 = pts[i][0], pts[i][1]
        x4, y4 = pts[i + 1][0], pts[i + 1][1]
        if _line_segments_intersect((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
            return True
    return False


def _find_minimum_bounding_box_route(
    points: List[Tuple[float, float]], padding_meters: float = 200.0
) -> Optional[Tuple[float, float, float, float, float]]:
    if len(points) < 2:
        return None
    padding = padding_meters
    min_area = float("inf")
    best_box = None
    for angle_deg in range(0, 181, 1):
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        cx = sum(x for x, y in points) / len(points)
        cy = sum(y for x, y in points) / len(points)
        rotated = [
            ((x - cx) * cos_a - (y - cy) * sin_a, (x - cx) * sin_a + (y - cy) * cos_a)
            for x, y in points
        ]
        xs, ys = [p[0] for p in rotated], [p[1] for p in rotated]
        w = max(xs) - min(xs) + 2 * padding
        h = max(ys) - min(ys) + 2 * padding
        if w * h < min_area:
            min_area = w * h
            best_box = (cx, cy, w, h, angle_rad)
    if best_box is None or best_box[2] <= 0 or best_box[3] <= 0:
        xs = [x for x, y in points]
        ys = [y for x, y in points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        best_box = (cx, cy, max(xs) - min(xs) + 2 * padding, max(ys) - min(ys) + 2 * padding, 0.0)
    return best_box


def _expand_box_to_include_all_points(
    points: List[Tuple[float, float]],
    center_x: float, center_y: float, width: float, height: float, angle: float,
    safety_margin: float = 100.0,
) -> Tuple[float, float, float, float, float]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = [
        ((x - center_x) * cos_a - (y - center_y) * sin_a, (x - center_x) * sin_a + (y - center_y) * cos_a)
        for x, y in points
    ]
    rxs, rys = [p[0] for p in rotated], [p[1] for p in rotated]
    min_rx, max_rx = min(rxs), max(rxs)
    min_ry, max_ry = min(rys), max(rys)
    req_w = (max_rx - min_rx) + 2 * safety_margin
    req_h = (max_ry - min_ry) + 2 * safety_margin
    center_rx = (min_rx + max_rx) / 2
    center_ry = (min_ry + max_ry) / 2
    cos_neg, sin_neg = math.cos(-angle), math.sin(-angle)
    new_cx = center_x + center_rx * cos_neg - center_ry * sin_neg
    new_cy = center_y + center_rx * sin_neg + center_ry * cos_neg
    return (new_cx, new_cy, max(width, req_w), max(height, req_h), angle)
