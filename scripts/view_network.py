#!/usr/bin/env python3
"""
Standalone script to display an exported SUMO network (.net.xml) and optional GPS trajectory.
Usage: python scripts/view_network.py <path-to-network.net.xml> [--trajectory-csv PATH] [--trajectory-num N]
Example: python scripts/view_network.py /path/to/config/porto.net.xml --trajectory-csv /path/to/train.csv --trajectory-num 1
"""

import argparse
import ast
import math
import os
import re
import sys
from typing import Optional


def trajectory_num_from_net_path(net_path: str) -> Optional[int]:
    """Extract trajectory number from network filename if it matches trajectory_<N>.net.xml."""
    basename = os.path.basename(net_path)
    m = re.match(r"trajectory_(\d+)\.net\.xml", basename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_trip_polyline(csv_path: str, trip_num: int):
    """Load polyline for a specific trip from CSV (Porto format: POLYLINE column as last column).
    Returns list of [lon, lat] points, or empty list on failure.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            next(f, None)  # skip header
            for i, line in enumerate(f, 1):
                if i == trip_num:
                    polyline_start = line.rfind('"[[')
                    if polyline_start == -1:
                        polyline_start = line.rfind('"[]')
                    if polyline_start != -1:
                        polyline_str = line[polyline_start + 1 :].strip().rstrip('"')
                        return ast.literal_eval(polyline_str)
                    return []
    except Exception as e:
        print(f"Error reading trajectory from CSV: {e}", file=sys.stderr)
    return []


def apply_trimming(polyline, static_threshold_m=15.0):
    """Trim static points at start/end using haversine distance threshold.
    polyline is list of [lon, lat] (Porto CSV POLYLINE format).
    """

    def haversine_m(coord1, coord2):
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


def _point_to_polyline_distance(px: float, py: float, shape_points) -> float:
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


def _project_point_onto_polyline(px: float, py: float, shape_points) -> tuple:
    """Closest point on the polyline to (px, py). Returns (proj_x, proj_y). Coords in SUMO."""
    if len(shape_points) < 2:
        if len(shape_points) == 1:
            return (float(shape_points[0][0]), float(shape_points[0][1]))
        return (px, py)
    best = (px, py)
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
    return best


def _project_point_onto_polyline_with_segment(px: float, py: float, shape_points) -> tuple:
    """Closest point on the polyline, segment index, and t in [0,1]. Returns (proj_x, proj_y, segment_index, t)."""
    if len(shape_points) < 2:
        if len(shape_points) == 1:
            return (float(shape_points[0][0]), float(shape_points[0][1]), 0, 0.0)
        return (px, py, 0, 0.0)
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
    return (best[0], best[1], best_seg, best_t)


def _build_edges_data(network_parser):
    """Build list of (edge_id, edge_data, shape_points) with shape in original SUMO coords."""
    edges_data = []
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


def draw_green_edges_for_segments(view, network_parser, sumo_points_flipped, top_per_segment=5):
    """For each trajectory segment, find matching network edges; draw top N in green. Returns (orange_edge_ids, green_edge_ids, start_edge_id, end_edge_id)."""
    if not sumo_points_flipped or len(sumo_points_flipped) < 2:
        return (set(), set(), None, None)
    y_min = getattr(view, "_network_y_min", 0)
    y_max = getattr(view, "_network_y_max", 0)

    def flip_y(y):
        return y_max + y_min - y

    edges_data = _build_edges_data(network_parser)
    if not edges_data:
        return (set(), set(), None, None)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPen

    direction_threshold = math.pi / 9
    angle_weight = 5.0
    orange_pen = QPen(QColor(255, 165, 0), 5)
    orange_pen.setStyle(Qt.SolidLine)
    green_pen = QPen(QColor(0, 255, 0), 4)
    green_pen.setStyle(Qt.SolidLine)

    all_orange_ids = set()
    all_green_ids = set()
    start_edge_id = None
    end_edge_id = None

    def draw_edge_shape(shape_points, pen, z_value):
        for i in range(len(shape_points) - 1):
            x1, y1 = shape_points[i][0], shape_points[i][1]
            x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
            y1_d = flip_y(y1)
            y2_d = flip_y(y2)
            line = view.scene.addLine(x1, y1_d, x2, y2_d, pen)
            line.setZValue(z_value)

    num_segments = len(sumo_points_flipped) - 1
    for seg_idx in range(num_segments):
        seg_x1, seg_y1 = sumo_points_flipped[seg_idx]
        seg_x2, seg_y2 = sumo_points_flipped[seg_idx + 1]
        seg_y1_sumo = flip_y(seg_y1)
        seg_y2_sumo = flip_y(seg_y2)

        dx_seg = seg_x2 - seg_x1
        dy_seg = seg_y2 - seg_y1
        segment_angle = math.atan2(dy_seg, dx_seg)

        matching = []
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

            d1 = _point_to_polyline_distance(seg_x1, seg_y1_sumo, shape_points)
            d2 = _point_to_polyline_distance(seg_x2, seg_y2_sumo, shape_points)
            score = angle_diff * angle_weight + (d1 + d2) / 2.0
            matching.append((score, edge_id, edge_data, shape_points, d1, d2))

            if d1 < closest_to_p1_dist:
                closest_to_p1_dist = d1
                closest_to_p1 = (edge_id, edge_data, shape_points)
            if d2 < closest_to_p2_dist:
                closest_to_p2_dist = d2
                closest_to_p2 = (edge_id, edge_data, shape_points)

        if closest_to_p1:
            eid, ed, sp = closest_to_p1
            all_orange_ids.add(eid)
            draw_edge_shape(sp, orange_pen, 6)
            if seg_idx == 0:
                start_edge_id = eid
        if closest_to_p2:
            eid, ed, sp = closest_to_p2
            all_orange_ids.add(eid)
            draw_edge_shape(sp, orange_pen, 6)
            if seg_idx == num_segments - 1:
                end_edge_id = eid

        matching.sort(key=lambda m: m[0])
        for _, eid, _, sp, _, _ in matching[:top_per_segment]:
            if eid not in all_orange_ids:
                all_green_ids.add(eid)
                draw_edge_shape(sp, green_pen, 5)

    if start_edge_id is None and sumo_points_flipped:
        px, py = sumo_points_flipped[0][0], flip_y(sumo_points_flipped[0][1])
        best_dist = float("inf")
        for edge_id, _, shape_points in edges_data:
            d = _point_to_polyline_distance(px, py, shape_points)
            if d < best_dist:
                best_dist = d
                start_edge_id = edge_id
    if end_edge_id is None and len(sumo_points_flipped) >= 2:
        px, py = sumo_points_flipped[-1][0], flip_y(sumo_points_flipped[-1][1])
        best_dist = float("inf")
        for edge_id, _, shape_points in edges_data:
            d = _point_to_polyline_distance(px, py, shape_points)
            if d < best_dist:
                best_dist = d
                end_edge_id = edge_id

    return (all_orange_ids, all_green_ids, start_edge_id, end_edge_id)


def shortest_path_dijkstra(network_parser, start_edge_id, end_edge_id, orange_ids, green_ids, roads_junctions_only=False):
    """Shortest path from start edge to end edge. Edge weights: orange=1, green=10, other=100. Returns list of edge_ids."""
    import heapq

    edges_dict = network_parser.get_edges()
    orange_ids = orange_ids or set()
    green_ids = green_ids or set()

    def weight(eid):
        if eid in orange_ids:
            return 1
        if eid in green_ids:
            return 10
        return 100

    adj = {}
    for eid, ed in edges_dict.items():
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

    heap = [(0, start_to, [])]
    seen = set()
    while heap:
        cost, node, path_edges = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == end_from:
            return [start_edge_id] + path_edges + [end_edge_id]
        for neighbor, eid, w in adj.get(node, []):
            if neighbor not in seen:
                heapq.heappush(heap, (cost + w, neighbor, path_edges + [eid]))
    return []


def build_directed_graph_scene(network_parser, roads_junctions_only=False, green_edge_ids=None, orange_edge_ids=None, path_edge_ids=None, trajectory_points=None):
    """Build a QGraphicsScene with the network as a directed graph. path_edge_ids: undashed red. trajectory_points: projected orange stars with red numbers (black bg)."""
    from PySide6.QtCore import Qt, QPointF, QRectF
    from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPainterPath, QPolygonF
    from PySide6.QtWidgets import (
        QGraphicsScene,
        QGraphicsEllipseItem,
        QGraphicsPathItem,
        QGraphicsLineItem,
        QGraphicsTextItem,
    )

    scene = QGraphicsScene()
    scene.setBackgroundBrush(QBrush(QColor(240, 242, 248)))

    conv = network_parser.conv_boundary
    bounds = network_parser.get_bounds()
    if conv:
        y_min, y_max = conv["y_min"], conv["y_max"]
    elif bounds:
        y_min, y_max = bounds["y_min"], bounds["y_max"]
    else:
        y_min, y_max = 0.0, 0.0

    def flip_y(y):
        return y_max + y_min - y

    junctions = network_parser.get_junctions()
    nodes_dict = network_parser.get_nodes()
    edges_dict = network_parser.get_edges()

    node_positions = {}
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

    def node_pos(nid):
        return node_positions.get(nid)

    all_x, all_y = [], []
    for x, y in node_positions.values():
        all_x.append(x)
        all_y.append(y)
    for ed in edges_dict.values():
        fp, tp = node_pos(ed.get("from")), node_pos(ed.get("to"))
        if fp:
            all_x.append(fp[0])
            all_y.append(fp[1])
        if tp:
            all_x.append(tp[0])
            all_y.append(tp[1])

    if not all_x or not all_y:
        no_items = QGraphicsTextItem("(No nodes/edges in graph)")
        no_items.setPos(10, 10)
        scene.addItem(no_items)
        scene.setSceneRect(QRectF(0, 0, 400, 80))
        return scene, QRectF(0, 0, 400, 80)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    range_x = max(1.0, max_x - min_x)
    range_y = max(1.0, max_y - min_y)
    padding = 50
    target_w, target_h = 700, 500
    scale = min(
        (target_w - 2 * padding) / range_x,
        (target_h - 2 * padding) / range_y,
    )

    def to_scene(x, y):
        return (padding + (x - min_x) * scale, padding + (y - min_y) * scale)

    node_radius = 0.625
    node_brush = QBrush(QColor(80, 120, 200))
    node_pen = QPen(Qt.PenStyle.NoPen)
    edge_pen = QPen(QColor(70, 70, 120), 0.16)
    edge_pen.setStyle(Qt.SolidLine)
    orange_edge_pen = QPen(QColor(255, 165, 0), 0.16)
    orange_edge_pen.setStyle(Qt.SolidLine)
    green_edge_pen = QPen(QColor(0, 255, 0), 0.16)
    green_edge_pen.setStyle(Qt.SolidLine)

    for jid in junctions:
        pos = node_positions.get(jid)
        if pos is None:
            continue
        x, y = pos
        sx, sy = to_scene(x, y)
        ellipse = QGraphicsEllipseItem(sx - node_radius, sy - node_radius, 2 * node_radius, 2 * node_radius)
        ellipse.setPen(node_pen)
        ellipse.setBrush(node_brush)
        ellipse.setZValue(0)
        scene.addItem(ellipse)

    path_set = set(path_edge_ids or [])
    path_pen = QPen(QColor(200, 0, 0), 0.24)
    path_pen.setStyle(Qt.SolidLine)

    path_points = []
    for i, eid in enumerate(path_edge_ids or []):
        ed = edges_dict.get(eid)
        if not ed:
            continue
        f, t = ed.get("from"), ed.get("to")
        pf, pt = node_pos(f), node_pos(t)
        if pf is None or pt is None:
            continue
        if i == 0:
            path_points.append(pf)
        path_points.append(pt)

    draw_full_path_edges = not (path_points and trajectory_points)

    arrow_len = 6.0
    arrow_wing = 2.2
    for eid, ed in edges_dict.items():
        if roads_junctions_only and not ed.get("allows_passenger", True):
            continue
        from_id, to_id = ed.get("from"), ed.get("to")
        fp = node_pos(from_id)
        tp = node_pos(to_id)
        if fp is None or tp is None:
            continue
        x1, y1 = fp
        x2, y2 = tp
        sx1, sy1 = to_scene(x1, y1)
        sx2, sy2 = to_scene(x2, y2)

        if eid in path_set:
            if not draw_full_path_edges:
                continue
            use_pen = path_pen
            z_line, z_arrow = 15, 16
        elif eid in (orange_edge_ids or set()):
            use_pen = orange_edge_pen
            z_line, z_arrow = 7, 8
        elif eid in (green_edge_ids or set()):
            use_pen = green_edge_pen
            z_line, z_arrow = 6, 7
        else:
            use_pen = edge_pen
            z_line, z_arrow = 5, 6

        use_arrow_brush = QBrush(use_pen.color())
        use_arrow_pen = QPen(Qt.PenStyle.NoPen)

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        inset = node_radius + arrow_len
        if length > 2 * inset:
            x2_line = x2 - ux * inset
            y2_line = y2 - uy * inset
        else:
            x2_line = (x1 + x2) / 2
            y2_line = (y1 + y2) / 2
        sx2l, sy2l = to_scene(x2_line, y2_line)
        line = scene.addLine(sx1, sy1, sx2l, sy2l, use_pen)
        line.setZValue(z_line)
        base_cx = x2 - ux * arrow_len
        base_cy = y2 - uy * arrow_len
        ax1 = base_cx + arrow_wing * (-uy)
        ay1 = base_cy + arrow_wing * ux
        ax2 = base_cx - arrow_wing * (-uy)
        ay2 = base_cy - arrow_wing * ux
        sax1, say1 = to_scene(ax1, ay1)
        sax2, say2 = to_scene(ax2, ay2)
        sx2_, sy2_ = to_scene(x2, y2)
        arrow = QPolygonF([QPointF(sx2_, sy2_), QPointF(sax1, say1), QPointF(sax2, say2)])
        path = QPainterPath()
        path.addPolygon(arrow)
        path_item = QGraphicsPathItem(path)
        path_item.setBrush(use_arrow_brush)
        path_item.setPen(use_arrow_pen)
        path_item.setZValue(z_arrow)
        scene.addItem(path_item)

    if path_points and trajectory_points:
        first_pt = trajectory_points[0]
        last_pt = trajectory_points[-1]
        fx, fy = _project_point_onto_polyline(first_pt[0], first_pt[1], path_points)
        lx, ly = _project_point_onto_polyline(last_pt[0], last_pt[1], path_points)
        _, _, i_first, _ = _project_point_onto_polyline_with_segment(first_pt[0], first_pt[1], path_points)
        _, _, i_last, _ = _project_point_onto_polyline_with_segment(last_pt[0], last_pt[1], path_points)
        if i_first < i_last:
            trimmed = [(fx, fy)] + [path_points[j] for j in range(i_first + 1, i_last + 1)] + [(lx, ly)]
        elif i_first == i_last:
            trimmed = [(fx, fy), (lx, ly)]
        else:
            trimmed = [(fx, fy), (lx, ly)]
        for k in range(len(trimmed) - 1):
            xa, ya = trimmed[k]
            xb, yb = trimmed[k + 1]
            if math.hypot(xb - xa, yb - ya) < 1e-9:
                continue
            sa1, sa2 = to_scene(xa, ya)
            sb1, sb2 = to_scene(xb, yb)
            line = scene.addLine(sa1, sa2, sb1, sb2, path_pen)
            line.setZValue(15)
        if len(trimmed) >= 2:
            xb, yb = trimmed[-1]
            xa, ya = trimmed[-2]
            dx, dy = xb - xa, yb - ya
            length = math.sqrt(dx * dx + dy * dy)
            if length >= 1e-6:
                ux, uy = dx / length, dy / length
                base_cx = xb - ux * arrow_len
                base_cy = yb - uy * arrow_len
                ax1 = base_cx + arrow_wing * (-uy)
                ay1 = base_cy + arrow_wing * ux
                ax2 = base_cx - arrow_wing * (-uy)
                ay2 = base_cy - arrow_wing * ux
                sax1, say1 = to_scene(ax1, ay1)
                sax2, say2 = to_scene(ax2, ay2)
                sx2_, sy2_ = to_scene(xb, yb)
                arrow_poly = QPolygonF([QPointF(sx2_, sy2_), QPointF(sax1, say1), QPointF(sax2, say2)])
                path_arrow = QPainterPath()
                path_arrow.addPolygon(arrow_poly)
                path_item = QGraphicsPathItem(path_arrow)
                path_item.setBrush(QBrush(path_pen.color()))
                path_item.setPen(QPen(Qt.PenStyle.NoPen))
                path_item.setZValue(16)
                scene.addItem(path_item)

    if path_points and trajectory_points:
        def create_star_path(cx, cy, radius):
            path = QPainterPath()
            num_points = 5
            outer, inner = radius, radius * 0.4
            for i in range(num_points * 2):
                angle = (i * math.pi) / num_points - math.pi / 2
                r = outer if i % 2 == 0 else inner
                x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.closeSubpath()
            return path

        star_radius = node_radius
        star_brush = QBrush(QColor(255, 165, 0))
        star_pen = QPen(QColor(255, 165, 0), 0.12)
        font = QFont()
        font.setPointSizeF(1.5)
        font.setBold(True)
        red_number_color = QColor(200, 0, 0)
        for idx, (px, py) in enumerate(trajectory_points):
            proj_x, proj_y = _project_point_onto_polyline(px, py, path_points)
            x, y = to_scene(proj_x, proj_y)
            star_path = create_star_path(x, y, star_radius)
            star_item = QGraphicsPathItem(star_path)
            star_item.setBrush(star_brush)
            star_item.setPen(star_pen)
            star_item.setZValue(20)
            scene.addItem(star_item)
            text_str = str(idx + 1)
            label_offset = star_radius
            tx, ty = x + label_offset, y - label_offset
            # Black outline of the number (small offsets in scene coords)
            outline_offset = 0.12
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                outline_item = QGraphicsTextItem(text_str)
                outline_item.setDefaultTextColor(QColor(0, 0, 0))
                outline_item.setFont(font)
                outline_item.setPos(tx + dx * outline_offset, ty + dy * outline_offset)
                outline_item.setZValue(20.5)
                scene.addItem(outline_item)
            text_item = QGraphicsTextItem(text_str)
            text_item.setDefaultTextColor(red_number_color)
            text_item.setFont(font)
            text_item.setPos(tx, ty)
            text_item.setZValue(21)
            scene.addItem(text_item)

    view_rect = QRectF(0, 0, target_w, target_h)
    scene.setSceneRect(view_rect)
    return scene, view_rect


def open_graph_window(network_parser, net_basename, roads_junctions_only=False, green_edge_ids=None, orange_edge_ids=None, path_edge_ids=None, trajectory_points=None, main_window=None):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QBrush, QColor, QWheelEvent
    from PySide6.QtWidgets import QApplication, QGraphicsView, QMainWindow

    scene, view_rect = build_directed_graph_scene(
        network_parser,
        roads_junctions_only=roads_junctions_only,
        green_edge_ids=green_edge_ids,
        orange_edge_ids=orange_edge_ids,
        path_edge_ids=path_edge_ids,
        trajectory_points=trajectory_points,
    )

    class GraphView(QGraphicsView):
        zoom_factor = 1.15
        min_zoom = 0.05
        max_zoom = 50.0

        def wheelEvent(self, event: QWheelEvent):
            if event.angleDelta().y() == 0:
                super().wheelEvent(event)
                return
            factor = self.zoom_factor if event.angleDelta().y() > 0 else 1.0 / self.zoom_factor
            self.scale(factor, factor)
            new_scale = self.transform().m11()
            if new_scale < self.min_zoom:
                self.scale(self.min_zoom / new_scale, self.min_zoom / new_scale)
            elif new_scale > self.max_zoom:
                self.scale(self.max_zoom / new_scale, self.max_zoom / new_scale)
            event.accept()

    graph_view = GraphView()
    graph_view.setScene(scene)
    graph_view.setDragMode(QGraphicsView.ScrollHandDrag)
    graph_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
    graph_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
    graph_view.setMinimumSize(400, 300)
    graph_view.setSceneRect(view_rect)
    graph_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
    graph_view.setBackgroundBrush(QBrush(QColor(225, 227, 232)))

    def do_fit():
        try:
            if graph_view.viewport().width() > 10 and graph_view.viewport().height() > 10:
                graph_view.resetTransform()
                graph_view.fitInView(view_rect, Qt.KeepAspectRatio)
                graph_view.viewport().update()
        except RuntimeError:
            pass  # window already closed

    graph_window = QMainWindow()
    graph_window.setWindowTitle(f"Directed graph — {net_basename}")
    graph_window.setCentralWidget(graph_view)
    graph_window.resize(800, 600)
    # Position to the right of the map window so both are visible
    if main_window is not None and main_window.isVisible():
        geom = main_window.frameGeometry()
        graph_window.move(geom.x() + geom.width() + 20, geom.y())
    else:
        graph_window.move(100, 100)
    graph_window.show()
    graph_window.raise_()
    graph_window.activateWindow()
    QTimer.singleShot(0, do_fit)
    QTimer.singleShot(50, do_fit)
    QTimer.singleShot(200, do_fit)
    return graph_window


def draw_trajectory_on_view(view, network_parser, polyline, trim=True, show_labels=True):
    """Draw GPS trajectory on the map view. Returns list of (x, y) sumo_points in display coords."""
    if trim:
        polyline = apply_trimming(polyline)
    if not polyline or len(polyline) < 2:
        print("Trajectory has too few points to draw.", file=sys.stderr)
        return None

    y_min = getattr(view, "_network_y_min", 0)
    y_max = getattr(view, "_network_y_max", 0)

    def flip_y(y):
        return y_max + y_min - y

    sumo_points = []
    for lon, lat in polyline:
        coords = network_parser.gps_to_sumo_coords(lon, lat)
        if coords:
            x, y = coords
            sumo_points.append((x, flip_y(y)))

    if len(sumo_points) < 2:
        print("Not enough trajectory points converted to SUMO coordinates.", file=sys.stderr)
        return None

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPainterPath
    from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsTextItem

    line_color = QColor(0, 100, 255)
    line_pen = QPen(line_color, 3)
    line_pen.setStyle(Qt.DashLine)
    for i in range(len(sumo_points) - 1):
        x1, y1 = sumo_points[i]
        x2, y2 = sumo_points[i + 1]
        line = view.scene.addLine(x1, y1, x2, y2, line_pen)
        line.setZValue(5)

    def create_star_path(cx, cy, radius):
        path = QPainterPath()
        num_points = 5
        outer, inner = radius, radius * 0.4
        for i in range(num_points * 2):
            angle = (i * math.pi) / num_points - math.pi / 2
            r = outer if i % 2 == 0 else inner
            x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    star_pen = QPen(QColor(255, 255, 0), 1.8)
    star_brush = QBrush(QColor(255, 255, 0))
    star_radius = 7.2
    font = QFont()
    font.setPointSize(7)
    font.setBold(True)

    for idx, (x, y) in enumerate(sumo_points):
        star_path = create_star_path(x, y, star_radius)
        star_item = QGraphicsPathItem(star_path)
        star_item.setPen(star_pen)
        star_item.setBrush(star_brush)
        star_item.setZValue(200)
        view.scene.addItem(star_item)

        if show_labels:
            text_str = str(idx + 1)
            label_offset = 7.2
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                outline = QGraphicsTextItem(text_str)
                outline.setPos(x + label_offset + dx, y - label_offset + dy)
                outline.setDefaultTextColor(QColor(0, 0, 0))
                outline.setFont(font)
                outline.setZValue(299)
                view.scene.addItem(outline)
            text_item = QGraphicsTextItem(text_str)
            text_item.setPos(x + label_offset, y - label_offset)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            text_item.setFont(font)
            text_item.setZValue(300)
            view.scene.addItem(text_item)

    print(f"Drew trajectory: {len(sumo_points)} points (trimmed={trim})")
    return sumo_points


def main():
    parser = argparse.ArgumentParser(description="View a SUMO network file (.net.xml) and optional GPS trajectory")
    parser.add_argument("network_file", nargs="?", default=None, help="Path to the .net.xml network file")
    parser.add_argument("--roads-only", action="store_true", help="Show only edges that allow passenger vehicles")
    parser.add_argument("--trajectory-csv", metavar="PATH", default=None, help="Path to CSV with POLYLINE column")
    parser.add_argument("--trajectory-num", type=int, default=None, metavar="N", help="Trajectory row number (1-based). Default from filename or 1")
    parser.add_argument("--no-trim", action="store_true", help="Do not trim static start/end points from trajectory")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw point number labels on trajectory")
    args = parser.parse_args()

    if not args.network_file:
        parser.print_help()
        print("\nExample:")
        print("  python scripts/view_network.py /path/to/config/porto.net.xml --trajectory-csv /path/to/train.csv --trajectory-num 1")
        sys.exit(1)

    net_path = os.path.abspath(args.network_file)
    if not os.path.isfile(net_path):
        print(f"Error: File not found: {net_path}", file=sys.stderr)
        sys.exit(1)

    if args.trajectory_csv and not os.path.isfile(os.path.abspath(args.trajectory_csv)):
        print(f"Error: Trajectory CSV not found: {args.trajectory_csv}", file=sys.stderr)
        sys.exit(1)

    if args.trajectory_csv:
        trajectory_num = args.trajectory_num
        if trajectory_num is None:
            from_name = trajectory_num_from_net_path(net_path)
            trajectory_num = from_name if from_name is not None else 1
            if from_name is not None:
                print(f"Using trajectory {trajectory_num} (from network filename)")
        args.trajectory_num = trajectory_num

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtWidgets import QApplication, QMainWindow
    from src.utils.network_parser import NetworkParser
    from src.gui.simulation_view import SimulationView

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    title = f"Network viewer — {os.path.basename(net_path)}"
    if args.trajectory_csv:
        title += f" (trajectory {args.trajectory_num})"
    window = QMainWindow()
    window.setWindowTitle(title)
    view = SimulationView()
    window.setCentralWidget(view)

    try:
        network_parser = NetworkParser(net_path)
        view.load_network(network_parser, roads_junctions_only=args.roads_only)
        view.set_osm_map_visible(False)
        edges = network_parser.get_edges()
        nodes = network_parser.get_nodes()
        junctions = network_parser.get_junctions()
        print(f"Loaded {len(edges)} edges, {len(nodes)} nodes, {len(junctions)} junctions")
    except Exception as e:
        print(f"Error loading network: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    orange_ids, green_ids = set(), set()
    path_edges = []
    sumo_points = []
    if args.trajectory_csv:
        polyline = load_trip_polyline(os.path.abspath(args.trajectory_csv), args.trajectory_num)
        if not polyline:
            print(f"Error: No trajectory found at row {args.trajectory_num}", file=sys.stderr)
        else:
            sumo_points = draw_trajectory_on_view(view, network_parser, polyline, trim=not args.no_trim, show_labels=not args.no_labels)
            if sumo_points:
                orange_ids, green_ids, start_edge_id, end_edge_id = draw_green_edges_for_segments(view, network_parser, sumo_points)
                if start_edge_id and end_edge_id:
                    path_edges = shortest_path_dijkstra(network_parser, start_edge_id, end_edge_id, orange_ids, green_ids, roads_junctions_only=args.roads_only)
                    if path_edges:
                        print(f"Shortest path: {len(path_edges)} edges")
                    else:
                        print("No route found from start edge to end edge.", file=sys.stderr)
                else:
                    print("Shortest path skipped: no start/end edge from trajectory.", file=sys.stderr)

    window.resize(1000, 700)
    window.show()

    # Keep reference so graph window is not garbage-collected
    graph_window = open_graph_window(
        network_parser,
        os.path.basename(net_path),
        roads_junctions_only=args.roads_only,
        green_edge_ids=green_ids,
        orange_edge_ids=orange_ids,
        path_edge_ids=path_edges,
        trajectory_points=sumo_points if (args.trajectory_csv and path_edges) else None,
        main_window=window,
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
