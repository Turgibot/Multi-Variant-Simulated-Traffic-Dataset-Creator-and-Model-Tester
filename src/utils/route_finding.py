"""
Shared route-finding logic: edge matching (orange/green), Dijkstra/A* shortest path.
Used by view_network.py and debug_trajectory_page.py.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

# Type alias for edges_data: list of (edge_id, edge_data, shape_points)
EdgesData = List[Tuple[str, Dict[str, Any], List[List[float]]]]


def apply_trimming(
    polyline: List[List[float]],
    static_threshold_m: float = 15.0,
) -> List[List[float]]:
    """Trim static points at start/end using haversine distance threshold.
    polyline is list of [lon, lat] (Porto CSV POLYLINE format).
    """
    def haversine_m(coord1: List[float], coord2: List[float]) -> float:
        lon1, lat1 = coord1[0], coord1[1]
        lon2, lat2 = coord2[0], coord2[1]
        R = 6371000
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

    if not polyline or len(polyline) < 2:
        return polyline

    real_start = 0
    for i in range(len(polyline) - 1):
        if haversine_m(polyline[i], polyline[i + 1]) > static_threshold_m:
            real_start = i
            break

    real_end = len(polyline) - 1
    for i in range(len(polyline) - 1, 0, -1):
        if haversine_m(polyline[i - 1], polyline[i]) > static_threshold_m:
            real_end = i
            break

    return polyline[real_start : real_end + 1]


def point_to_polyline_distance(px: float, py: float, shape_points: List[List[float]]) -> float:
    """Minimum distance from point (px, py) to any segment of the polyline. Coords in SUMO."""
    if len(shape_points) < 2:
        if len(shape_points) == 1:
            dx = px - shape_points[0][0]
            dy = py - shape_points[0][1]
            return math.sqrt(dx * dx + dy * dy)
        return float("inf")
    min_dist = float("inf")
    for i in range(len(shape_points) - 1):
        x1, y1 = shape_points[i][0], shape_points[i][1]
        x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq == 0:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx, cy = x1 + t * seg_dx, y1 + t * seg_dy
            dist = math.hypot(px - cx, py - cy)
        min_dist = min(min_dist, dist)
    return min_dist


def project_point_onto_polyline(
    px: float, py: float, shape_points: List[List[float]]
) -> Tuple[float, float]:
    """Closest point on the polyline to (px, py). Returns (proj_x, proj_y)."""
    result, _ = project_point_onto_polyline_with_segment(px, py, shape_points)
    return result


def project_point_onto_polyline_with_segment(
    px: float, py: float, shape_points: List[List[float]]
) -> Tuple[Tuple[float, float], int]:
    """Closest point on the polyline to (px, py). Returns ((proj_x, proj_y), segment_idx).
    segment_idx is the index of the segment (between shape_points[i] and shape_points[i+1])."""
    if len(shape_points) < 2:
        if len(shape_points) == 1:
            return ((float(shape_points[0][0]), float(shape_points[0][1])), 0)
        return ((px, py), 0)
    best = (px, py)
    best_seg = 0
    best_dist_sq = float("inf")
    for i in range(len(shape_points) - 1):
        x1, y1 = shape_points[i][0], shape_points[i][1]
        x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq == 0:
            cx, cy = x1, y1
        else:
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx, cy = x1 + t * seg_dx, y1 + t * seg_dy
        d_sq = (px - cx) ** 2 + (py - cy) ** 2
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best = (cx, cy)
            best_seg = i
    return (best, best_seg)


def project_point_onto_polyline_with_segment_and_t(
    px: float, py: float, shape_points: List[List[float]]
) -> Tuple[Tuple[float, float], int, float]:
    """Closest point on the polyline to (px, py). Returns ((proj_x, proj_y), segment_idx, t).
    t is the fractional position along the segment (0=start, 1=end)."""
    if len(shape_points) < 2:
        if len(shape_points) == 1:
            return ((float(shape_points[0][0]), float(shape_points[0][1])), 0, 0.0)
        return ((px, py), 0, 0.0)
    best = (px, py)
    best_seg = 0
    best_t = 0.0
    best_dist_sq = float("inf")
    for i in range(len(shape_points) - 1):
        x1, y1 = shape_points[i][0], shape_points[i][1]
        x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq == 0:
            cx, cy = x1, y1
            t = 0.0
        else:
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx, cy = x1 + t * seg_dx, y1 + t * seg_dy
        d_sq = (px - cx) ** 2 + (py - cy) ** 2
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best = (cx, cy)
            best_seg = i
            best_t = t
    return (best, best_seg, best_t)


def build_edges_data(network_parser: Any) -> EdgesData:
    """Build list of (edge_id, edge_data, shape_points) with shape in original SUMO coords.
    Includes all edges from the network.
    """
    edges_data: EdgesData = []
    for edge_id, edge_data in network_parser.get_edges().items():
        lanes = edge_data.get("lanes", [])
        if not lanes:
            continue
        shape = lanes[0].get("shape", [])
        if len(shape) < 2:
            continue
        shape_points = [[float(p[0]), float(p[1])] for p in shape]
        edges_data.append((edge_id, edge_data, shape_points))
    return edges_data


def compute_green_orange_edges(
    edges_data: EdgesData,
    sumo_points_flipped: List[Tuple[float, float]],
    y_min: float,
    y_max: float,
    top_per_segment: int = 5,
    direction_threshold: float = math.pi / 9,
    angle_weight: float = 5.0,
    spatial_index: Optional[Any] = None,
    filter_radius: float = 800.0,
) -> Tuple[Set[str], Set[str], Optional[str], Optional[str], List[str]]:
    """Compute orange/green edge IDs and start/end edge for route finding (no drawing).
    sumo_points_flipped: trajectory points in display coords (Y-flipped).
    Returns (orange_edge_ids, green_edge_ids, start_edge_id, end_edge_id, start_edge_candidates).
    Same logic as view_network draw_green_edges_for_segments.
    When spatial_index is provided, filters edges to candidates near trajectory for speed.
    """
    if not sumo_points_flipped or len(sumo_points_flipped) < 2:
        return (set(), set(), None, None, [])

    def flip_y(y: float) -> float:
        return y_max + y_min - y

    # Optional: filter edges by spatial index for faster processing
    if spatial_index is not None:
        candidate_ids: Set[str] = set()
        for px, py in sumo_points_flipped:
            py_sumo = flip_y(py)
            candidate_ids.update(spatial_index.get_candidates_in_radius(px, py_sumo, filter_radius))
        edge_id_set = {eid for eid, _, _ in edges_data}
        candidate_ids &= edge_id_set
        if candidate_ids:
            edges_data = [(eid, ed, sp) for eid, ed, sp in edges_data if eid in candidate_ids]
        if not edges_data:
            return (set(), set(), None, None, [])

    all_orange_ids: Set[str] = set()
    all_green_ids: Set[str] = set()
    start_edge_id: Optional[str] = None
    end_edge_id: Optional[str] = None
    start_edge_candidates: List[str] = []

    num_segments = len(sumo_points_flipped) - 1
    for seg_idx in range(num_segments):
        seg_x1, seg_y1 = sumo_points_flipped[seg_idx]
        seg_x2, seg_y2 = sumo_points_flipped[seg_idx + 1]
        seg_y1_sumo = flip_y(seg_y1)
        seg_y2_sumo = flip_y(seg_y2)

        dx_seg = seg_x2 - seg_x1
        dy_seg = seg_y2 - seg_y1
        segment_angle = math.atan2(dy_seg, dx_seg)

        matching: List[Tuple[float, str, Any, List[List[float]], float, float]] = []
        closest_to_p1 = None
        closest_to_p1_dist = float("inf")
        closest_to_p2 = None
        closest_to_p2_dist = float("inf")

        for edge_id, edge_data, shape_points in edges_data:
            if len(shape_points) < 2:
                continue
            x_start, y_start = shape_points[0][0], shape_points[0][1]
            x_end, y_end = shape_points[-1][0], shape_points[-1][1]
            y_start_f = flip_y(y_start)
            y_end_f = flip_y(y_end)
            dx_edge = x_end - x_start
            dy_edge = y_end_f - y_start_f
            edge_angle = math.atan2(dy_edge, dx_edge)
            angle_diff = abs(segment_angle - edge_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > direction_threshold:
                continue

            d1 = point_to_polyline_distance(seg_x1, seg_y1_sumo, shape_points)
            d2 = point_to_polyline_distance(seg_x2, seg_y2_sumo, shape_points)
            score = angle_diff * angle_weight + (d1 + d2) / 2.0
            matching.append((score, edge_id, edge_data, shape_points, d1, d2))

            if d1 < closest_to_p1_dist:
                closest_to_p1_dist = d1
                closest_to_p1 = (edge_id, edge_data, shape_points)
            if d2 < closest_to_p2_dist:
                closest_to_p2_dist = d2
                closest_to_p2 = (edge_id, edge_data, shape_points)

        if closest_to_p1:
            eid, _ed, _sp = closest_to_p1
            all_orange_ids.add(eid)
            if seg_idx == 0:
                start_edge_id = eid
                start_edge_candidates = [m[1] for m in sorted(matching, key=lambda m: m[4])[:5]]
        if closest_to_p2:
            eid, _ed, _sp = closest_to_p2
            all_orange_ids.add(eid)
            if seg_idx == num_segments - 1:
                end_edge_id = eid

        matching.sort(key=lambda m: m[0])
        for _, eid, _, _sp, _, _ in matching[:top_per_segment]:
            if eid not in all_orange_ids:
                all_green_ids.add(eid)

    # Fallbacks if start/end not set from segments
    if start_edge_id is None and sumo_points_flipped:
        px, py = sumo_points_flipped[0][0], flip_y(sumo_points_flipped[0][1])
        by_dist: List[Tuple[float, str]] = []
        for edge_id, _ed, shape_points in edges_data:
            d = point_to_polyline_distance(px, py, shape_points)
            by_dist.append((d, edge_id))
        by_dist.sort(key=lambda x: x[0])
        start_edge_candidates = [eid for _, eid in by_dist[:5]]
        start_edge_id = start_edge_candidates[0] if start_edge_candidates else None
    if end_edge_id is None and len(sumo_points_flipped) >= 2:
        px, py = sumo_points_flipped[-1][0], flip_y(sumo_points_flipped[-1][1])
        best_dist = float("inf")
        for edge_id, _ed, shape_points in edges_data:
            d = point_to_polyline_distance(px, py, shape_points)
            if d < best_dist:
                best_dist = d
                end_edge_id = edge_id

    if start_edge_id and not start_edge_candidates:
        start_edge_candidates = [start_edge_id]
    return (all_orange_ids, all_green_ids, start_edge_id, end_edge_id, start_edge_candidates)


def build_node_positions(network_parser: Any) -> Dict[str, Tuple[float, float]]:
    """Build node_id -> (x, y) in same coords as trajectory (flip_y applied). For A* heuristic."""
    conv = network_parser.conv_boundary
    bounds = network_parser.get_bounds()
    if conv:
        y_min, y_max = conv["y_min"], conv["y_max"]
    elif bounds:
        y_min, y_max = bounds["y_min"], bounds["y_max"]
    else:
        y_min, y_max = 0.0, 0.0

    def flip_y(y: float) -> float:
        return y_max + y_min - y

    junctions = network_parser.get_junctions()
    nodes_dict = network_parser.get_nodes()
    edges_dict = network_parser.get_edges()
    node_positions: Dict[str, Tuple[float, float]] = {}
    for jid, j in junctions.items():
        node_positions[jid] = (j["x"], flip_y(j["y"]))
    for nid, n in nodes_dict.items():
        if nid not in node_positions:
            node_positions[nid] = (n["x"], flip_y(n["y"]))
    for eid, ed in edges_dict.items():
        from_id, to_id = ed.get("from"), ed.get("to")
        lanes = ed.get("lanes", [])
        if not lanes:
            continue
        shape = lanes[0].get("shape", [])
        if len(shape) < 2:
            continue
        if from_id and from_id not in node_positions:
            node_positions[from_id] = (shape[0][0], flip_y(shape[0][1]))
        if to_id and to_id not in node_positions:
            node_positions[to_id] = (shape[-1][0], flip_y(shape[-1][1]))
    return node_positions


def shortest_path_dijkstra(
    network_parser: Any,
    start_edge_id: str,
    end_edge_id: str,
    orange_ids: Optional[Set[str]] = None,
    green_ids: Optional[Set[str]] = None,
    node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    goal_xy: Optional[Tuple[float, float]] = None,
    edges_in_polygon: Optional[Set[str]] = None,
) -> List[str]:
    """Shortest path from start edge to end edge. Edge weights: orange=1, green=10, other=1000.
    If edges_in_polygon is provided, only edges inside the polygon are used.
    If node_positions and goal_xy are provided, uses A* with heuristic = distance to goal (admissible).
    Returns list of edge_ids.
    """
    import heapq

    edges_dict = network_parser.get_edges()
    orange_ids = orange_ids or set()
    green_ids = green_ids or set()

    def weight(eid: str) -> float:
        if eid in orange_ids:
            return 1
        if eid in green_ids:
            return 10
        return 1000

    adj: Dict[str, List[Tuple[str, str, float]]] = {}
    for eid, ed in edges_dict.items():
        if edges_in_polygon is not None and eid not in edges_in_polygon:
            continue
        from_id = ed.get("from")
        to_id = ed.get("to")
        if not from_id or not to_id:
            continue
        w = weight(eid)
        adj.setdefault(from_id, []).append((to_id, eid, w))

    if start_edge_id not in edges_dict or end_edge_id not in edges_dict:
        return []
    if start_edge_id == end_edge_id:
        return [start_edge_id]
    start_to = edges_dict[start_edge_id].get("to")
    end_from = edges_dict[end_edge_id].get("from")
    if start_to is None or end_from is None:
        return []

    use_astar = node_positions is not None and goal_xy is not None and len(node_positions) > 0
    max_edge_length = 1.0
    if use_astar:
        for eid, ed in edges_dict.items():
            fp = node_positions.get(ed.get("from"))
            tp = node_positions.get(ed.get("to"))
            if fp and tp:
                d = math.hypot(tp[0] - fp[0], tp[1] - fp[1])
                if d > max_edge_length:
                    max_edge_length = d
        max_edge_length = max(max_edge_length, 1.0)
        min_cost_per_meter = 1.0 / max_edge_length

        def heuristic(nid: str) -> float:
            pos = node_positions.get(nid)
            if pos is None:
                return 0.0
            d = math.hypot(goal_xy[0] - pos[0], goal_xy[1] - pos[1])
            return d * min_cost_per_meter

    if use_astar:
        h0 = heuristic(start_to)
        heap: List[Tuple[float, float, str, List[str]]] = [(0 + h0, 0, start_to, [])]
    else:
        heap = [(0, start_to, [])]

    seen: Set[str] = set()
    while heap:
        if use_astar:
            f, g, node, path_edges = heapq.heappop(heap)
        else:
            cost, node, path_edges = heapq.heappop(heap)
            g = cost
        if node in seen:
            continue
        seen.add(node)
        if node == end_from:
            return [start_edge_id] + path_edges + [end_edge_id]
        for neighbor, eid, w in adj.get(node, []):
            if neighbor not in seen:
                g_new = g + w
                if use_astar:
                    h_new = heuristic(neighbor)
                    f_new = g_new + h_new
                    heapq.heappush(heap, (f_new, g_new, neighbor, path_edges + [eid]))
                else:
                    heapq.heappush(heap, (g_new, neighbor, path_edges + [eid]))
    return []
