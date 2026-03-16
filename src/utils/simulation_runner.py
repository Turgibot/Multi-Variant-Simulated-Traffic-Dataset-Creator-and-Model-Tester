"""
Simulation loop: update (sync DB from TraCI) and dispatch (add scheduled vehicles to SUMO).

Legacy order per iteration: traci.simulationStep() -> update(step, traci) -> dispatch(step, traci).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.simulation_db import SimulationDB


class SimulationRunner:
    """Runs update and dispatch against the simulation DB and TraCI."""

    def __init__(self, project_path: str, project_name: str):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.sim_db = SimulationDB(project_path, project_name)

    def update(self, current_step: int, traci: Any) -> None:
        """
        Sync vehicle and road state from TraCI to DB (legacy SimManager.update).

        For each vehicle in TraCI that we track (in_route in DB), update position,
        speed, current_edge, route_left; update road vehicles_on_road; detect arrival.
        """
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except Exception:
            return

        with self.sim_db.connect() as conn:
            for vid in vehicle_ids:
                row = conn.execute(
                    "SELECT id, status, current_edge, destination_edge FROM vehicles WHERE id = ?",
                    (vid,),
                ).fetchone()
                if not row or str(row[1]) != "in_route":
                    continue

                try:
                    current_edge = traci.vehicle.getRoadID(vid)
                    current_position = float(traci.vehicle.getLanePosition(vid))
                    speed = float(traci.vehicle.getSpeed(vid))
                    acceleration = float(traci.vehicle.getAcceleration(vid))
                    x, y = traci.vehicle.getPosition(vid)
                except Exception:
                    continue

                prev_edge = str(row[2]) if row[2] else None
                destination_edge = str(row[3]) if row[3] else None

                # Road zone for current edge
                road_row = conn.execute(
                    "SELECT id, zone, vehicles_on_road_json FROM roads WHERE id = ?",
                    (current_edge,),
                ).fetchone()
                current_zone = str(road_row[1]) if road_row and road_row[1] else None

                # Route left: get current route from DB and trim to current edge onward
                route_row = conn.execute(
                    "SELECT route_json FROM vehicles WHERE id = ?", (vid,)
                ).fetchone()
                route_left = []
                if route_row and route_row[0]:
                    try:
                        full_route = json.loads(route_row[0])
                        if isinstance(full_route, list) and current_edge in full_route:
                            idx = full_route.index(current_edge)
                            route_left = full_route[idx:]
                    except Exception:
                        pass

                arrived = current_edge == destination_edge and destination_edge

                if arrived:
                    conn.execute(
                        """
                        UPDATE vehicles SET
                            status = ?, current_edge = ?, current_position = ?, speed = ?, acceleration = ?,
                            current_x = ?, current_y = ?, current_zone = ?,
                            route_left_json = ?, route_length_left = 0,
                            destination_step = ?
                        WHERE id = ?
                        """,
                        (
                            "parked",
                            current_edge,
                            current_position,
                            speed,
                            acceleration,
                            float(x),
                            float(y),
                            current_zone,
                            json.dumps([], ensure_ascii=False),
                            current_step,
                            vid,
                        ),
                    )
                    # Remove from current road's vehicles_on_road
                    if road_row and road_row[2]:
                        try:
                            on_road = json.loads(road_row[2])
                            if isinstance(on_road, list) and vid in on_road:
                                on_road = [v for v in on_road if v != vid]
                                conn.execute(
                                    "UPDATE roads SET vehicles_on_road_json = ? WHERE id = ?",
                                    (json.dumps(on_road, ensure_ascii=False), current_edge),
                                )
                        except Exception:
                            pass
                else:
                    route_length_left = 0.0
                    if route_left:
                        placeholders = ",".join("?" for _ in route_left)
                        length_rows = conn.execute(
                            f"SELECT id, length FROM roads WHERE id IN ({placeholders})",
                            route_left,
                        ).fetchall()
                        route_length_left = sum(float(r[1]) for r in length_rows)

                    conn.execute(
                        """
                        UPDATE vehicles SET
                            current_edge = ?, current_position = ?, speed = ?, acceleration = ?,
                            current_x = ?, current_y = ?, current_zone = ?,
                            route_left_json = ?, route_length_left = ?
                        WHERE id = ?
                        """,
                        (
                            current_edge,
                            current_position,
                            speed,
                            acceleration,
                            float(x),
                            float(y),
                            current_zone,
                            json.dumps(route_left, ensure_ascii=False),
                            route_length_left,
                            vid,
                        ),
                    )

                    # Update current road: ensure vehicle is in vehicles_on_road
                    if road_row and road_row[2]:
                        try:
                            on_road = json.loads(road_row[2])
                            if not isinstance(on_road, list):
                                on_road = []
                            if vid not in on_road:
                                on_road.append(vid)
                                conn.execute(
                                    "UPDATE roads SET vehicles_on_road_json = ? WHERE id = ?",
                                    (json.dumps(on_road, ensure_ascii=False), current_edge),
                                )
                        except Exception:
                            pass

                    # Remove from previous road if changed
                    if prev_edge and prev_edge != current_edge:
                        prev_road = conn.execute(
                            "SELECT vehicles_on_road_json FROM roads WHERE id = ?",
                            (prev_edge,),
                        ).fetchone()
                        if prev_road and prev_road[0]:
                            try:
                                prev_on_road = json.loads(prev_road[0])
                                if isinstance(prev_on_road, list) and vid in prev_on_road:
                                    prev_on_road = [v for v in prev_on_road if v != vid]
                                    conn.execute(
                                        "UPDATE roads SET vehicles_on_road_json = ? WHERE id = ?",
                                        (
                                            json.dumps(prev_on_road, ensure_ascii=False),
                                            prev_edge,
                                        ),
                                    )
                            except Exception:
                                pass

            conn.commit()

    def dispatch(self, current_step: int, traci: Any) -> None:
        """
        Dispatch all vehicles scheduled for this step (legacy SimManager.dispatch).

        Reads vehicle_schedule_assignments for current_step, adds each vehicle to SUMO
        with route from origin to destination, updates DB.
        """
        try:
            import traci.constants as tc
        except ImportError:
            tc = None

        with self.sim_db.connect() as conn:
            rows = conn.execute(
                """
                SELECT vehicle_id, origin_name, destination_name
                FROM vehicle_schedule_assignments
                WHERE simulation_step = ?
                """,
                (current_step,),
            ).fetchall()
            # DEBUG: log every scheduled row read for this timestep.
            # for r in rows:
            #     try:
            #         print(
            #             f"[DEBUG] dispatch read: step={current_step}, vehicle_id={r[0]}, "
            #             f"origin_name={r[1]}, destination_name={r[2]}"
            #         )
            #     except Exception:
            #         pass

            for row in rows:
                vehicle_id = str(row[0])
                origin_label = str(row[1])
                destination_label = str(row[2])

                veh_row = conn.execute(
                    "SELECT status, vehicle_type, destinations_json, current_edge, current_position FROM vehicles WHERE id = ?",
                    (vehicle_id,),
                ).fetchone()
                if not veh_row or str(veh_row[0]) != "parked":
                    # print(f"[WARNING] dispatch no veh_row for {vehicle_id} vehicle not parked")
                    continue

                try:
                    dest_json = json.loads(veh_row[2]) if veh_row[2] else {}
                except Exception:
                    # print(f"[WARNING] dispatch no destinations_json for {vehicle_id} error: {e}")
                    continue
                if not isinstance(dest_json, dict):
                    # print(f"[WARNING] dispatch no destinations_json for {vehicle_id}")
                    continue

                origin = dest_json.get(origin_label) if isinstance(dest_json.get(origin_label), dict) else None
                destination = dest_json.get(destination_label) if isinstance(dest_json.get(destination_label), dict) else None
                if not origin or not destination or "edge" not in origin or "edge" not in destination:
                    continue

                origin_edge = str(origin["edge"])
                origin_position = float(origin.get("position", 0.0))
                dest_edge = str(destination["edge"])
                dest_position = float(destination.get("position", 0.0))
                # print(f"[DEBUG] dispatch origin_edge: {origin_edge}, origin_position: {origin_position}, dest_edge: {dest_edge}, dest_position: {dest_position}")

                try:
                    route_result = traci.simulation.findRoute(origin_edge, dest_edge)
                    full_route_edges = list(route_result.edges)
                except Exception:
                    continue
                if not full_route_edges:
                    # print(f"[WARNING] dispatch no full_route_edges for {vehicle_id}")
                    continue

                route_id = f"route_{vehicle_id}_to_{destination_label}_{current_step}"
                try:
                    traci.route.add(routeID=route_id, edges=full_route_edges)
                except Exception:
                    # print(f"[WARNING] dispatch no route_id for {vehicle_id}")
                    continue

                depart = current_step + 1
                try:
                    traci.vehicle.add(
                        vehID=vehicle_id,
                        routeID=route_id,
                        typeID=str(veh_row[1]),
                        depart=float(depart),
                        departPos=origin_position,
                        departSpeed=0,
                        departLane="0",
                    )
                except Exception:
                    continue

                is_noise_row = conn.execute(
                    "SELECT is_noise, color FROM vehicles WHERE id = ?", (vehicle_id,)
                ).fetchone()
                if is_noise_row and is_noise_row[0]:
                    try:
                        traci.vehicle.setColor(vehicle_id, (255, 255, 255))
                    except Exception:
                        pass

                try:
                    ox, oy = traci.simulation.convert2D(origin_edge, origin_position, 0)
                    dx, dy = traci.simulation.convert2D(dest_edge, dest_position, 0)
                except Exception:
                    ox = oy = dx = dy = 0.0

                road_zone_row = conn.execute(
                    "SELECT zone FROM roads WHERE id = ?", (origin_edge,)
                ).fetchone()
                origin_zone = str(road_zone_row[0]) if road_zone_row and road_zone_row[0] else None
                dest_zone_row = conn.execute(
                    "SELECT zone FROM roads WHERE id = ?", (dest_edge,)
                ).fetchone()
                dest_zone = str(dest_zone_row[0]) if dest_zone_row and dest_zone_row[0] else None

                route_length = 0.0
                for e in full_route_edges:
                    r = conn.execute("SELECT length FROM roads WHERE id = ?", (e,)).fetchone()
                    if r:
                        route_length += float(r[0])

                conn.execute(
                    """
                    UPDATE vehicles SET
                        status = ?, current_edge = ?, current_position = ?,
                        origin_name = ?, origin_edge = ?, origin_position = ?, origin_zone = ?,
                        origin_x = ?, origin_y = ?, origin_start_sec = ?,
                        destination_name = ?, destination_edge = ?, destination_position = ?, destination_zone = ?,
                        destination_x = ?, destination_y = ?,
                        route_json = ?, route_length = ?, route_left_json = ?, route_length_left = ?,
                        current_x = ?, current_y = ?, current_zone = ?
                    WHERE id = ?
                    """,
                    (
                        "in_route",
                        origin_edge,
                        origin_position,
                        origin_label,
                        origin_edge,
                        origin_position,
                        origin_zone,
                        float(ox),
                        float(oy),
                        depart,
                        destination_label,
                        dest_edge,
                        dest_position,
                        dest_zone,
                        float(dx),
                        float(dy),
                        json.dumps(full_route_edges, ensure_ascii=False),
                        route_length,
                        json.dumps(full_route_edges, ensure_ascii=False),
                        route_length,
                        float(ox),
                        float(oy),
                        origin_zone,
                        vehicle_id,
                    ),
                )

                # Add vehicle to origin road's vehicles_on_road
                road_row = conn.execute(
                    "SELECT vehicles_on_road_json FROM roads WHERE id = ?", (origin_edge,)
                ).fetchone()
                if road_row and road_row[0] is not None:
                    try:
                        on_road = json.loads(road_row[0])
                        if not isinstance(on_road, list):
                            on_road = []
                        if vehicle_id not in on_road:
                            on_road.append(vehicle_id)
                            conn.execute(
                                "UPDATE roads SET vehicles_on_road_json = ? WHERE id = ?",
                                (json.dumps(on_road, ensure_ascii=False), origin_edge),
                            )
                    except Exception:
                        pass

            conn.commit()

    def get_vehicle_display_color(self, vehicle_id: str) -> Tuple[int, int, int]:
        """
        Return (r, g, b) 0-255 for a vehicle for display. Noise vehicles are black.
        Falls back to vehicle_types.color when vehicles.color is null/empty.
        """
        with self.sim_db.connect() as conn:
            row = conn.execute(
                "SELECT color, is_noise, vehicle_type FROM vehicles WHERE id = ?",
                (vehicle_id,),
            ).fetchone()
            if not row:
                return (128, 128, 128)  # default gray
            color_str, is_noise, vehicle_type = str(row[0] or "").strip(), bool(row[1]), (row[2] or "")
            if is_noise:
                return (0, 0, 0)
            if not color_str and vehicle_type:
                type_row = conn.execute(
                    "SELECT color FROM vehicle_types WHERE type_name = ?",
                    (vehicle_type,),
                ).fetchone()
                if type_row and type_row[0]:
                    color_str = str(type_row[0]).strip()
            if not color_str:
                return (128, 128, 128)
            return _parse_color_to_rgb(color_str)


def _parse_color_to_rgb(color_str: str) -> Tuple[int, int, int]:
    """Parse color string (hex, 'r,g,b', or name) to (r, g, b) 0-255."""
    s = color_str.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 6:
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                pass
        elif len(s) == 3:
            try:
                return (int(s[0] * 2, 16), int(s[1] * 2, 16), int(s[2] * 2, 16))
            except ValueError:
                pass
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            try:
                return (
                    max(0, min(255, int(parts[0]))),
                    max(0, min(255, int(parts[1]))),
                    max(0, min(255, int(parts[2]))),
                )
            except (ValueError, TypeError):
                pass
    _names = {
        "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
        "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "gray": (128, 128, 128), "grey": (128, 128, 128), "orange": (255, 165, 0),
        "cyan": (0, 255, 255), "magenta": (255, 0, 255), "brown": (165, 42, 42),
    }
    return _names.get(s.lower(), (128, 128, 128))
