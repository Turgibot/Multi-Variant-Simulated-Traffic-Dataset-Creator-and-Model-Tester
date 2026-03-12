"""
Simulation DB service (step 2 scope).

Implemented in this step:
- Load + validate simulation.config.json
- Load SUMO network parser instance

Not implemented here yet:
- Entity/scheduler preparation
- DB writes
- Runtime loop integration
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.network_parser import NetworkParser
from src.utils.simulation_db import SimulationDB, compute_config_hash
from src.utils.sumo_config_manager import SUMOConfigManager


_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class SimulationDBService:
    """Service layer for preparation preconditions and inputs."""

    def __init__(self, project_name: str, project_path: str):
        self.project_name = project_name
        self.project_path = Path(project_path)
        self.simulation_config_path = self.project_path / "simulation.config.json"
        self.config_manager = SUMOConfigManager(project_path)
        self.sim_db = SimulationDB(project_path, project_name)

    def load_and_validate_config(self, config_path: Optional[str] = None) -> Tuple[Dict, str]:
        """
        Load simulation config JSON and validate required structure/rules.

        Returns:
            (config_dict, config_hash)
        """
        path = Path(config_path) if config_path else self.simulation_config_path
        if not path.exists():
            raise ValueError(f"Simulation config not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in simulation config: {exc.msg} (line {exc.lineno}, col {exc.colno})"
            ) from exc

        if not isinstance(config, dict):
            raise ValueError("Simulation config root must be a JSON object.")

        self._validate_config(config)
        return config, compute_config_hash(config)

    def load_network(self, net_path: Optional[str] = None) -> Tuple[NetworkParser, Path]:
        """
        Load network parser for preparation.

        Network path resolution order:
        1) explicit net_path argument
        2) parsed network file from project's configured sumocfg
        """
        resolved_net_path: Optional[Path] = None
        if net_path:
            resolved_net_path = Path(net_path).resolve()
        else:
            config_files = self.config_manager.get_config_files()
            network_path = config_files.get("network")
            if network_path:
                resolved_net_path = Path(network_path).resolve()

        if resolved_net_path is None:
            raise ValueError("Network path not found. Set a valid SUMO .sumocfg first.")
        if not resolved_net_path.exists():
            raise ValueError(f"Network file does not exist: {resolved_net_path}")

        parser = NetworkParser(str(resolved_net_path))
        return parser, resolved_net_path

    def prepare_initial_snapshot(
        self,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Step A: build initial simulation DB snapshot using config + network.

        Populates:
        - static entities: junctions, roads, zones, landmarks, vehicle_types
        - initial vehicles based on zone allocation and type distribution
        - schedule windows from weekday_schedule
        - metadata fingerprint (config hash)
        """
        def emit(pct: int, msg: str) -> None:
            if progress_cb:
                progress_cb(pct, msg)

        def cancelled() -> bool:
            return bool(is_cancelled and is_cancelled())

        try:
            emit(5, "Loading and validating simulation config...")
            config, _config_hash = self.load_and_validate_config()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(15, "Loading SUMO network...")
            parser, net_path = self.load_network()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(25, "Initializing simulation database...")
            self.sim_db.initialize_schema(overwrite=True)
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(35, "Building zone and landmark mappings...")
            zone_edge_map, zone_node_map = self._build_zone_maps(parser)
            landmarks_map = self._build_landmark_map(config)
            positive_zones = self._get_positive_zones_from_config(config)
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(50, "Writing static entities...")
            with self.sim_db.connect() as conn:
                self._insert_junctions(conn, parser, zone_node_map)
                self._insert_roads(conn, parser, zone_edge_map)
                self._insert_zones(conn, config, zone_edge_map, zone_node_map)
                self._insert_landmarks(conn, landmarks_map)
                self._insert_vehicle_types(conn, config)
                self._insert_schedule_windows(conn, config, positive_zones)
                conn.commit()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(70, "Generating initial vehicles...")
            with self.sim_db.connect() as conn:
                vehicles_count, required_vehicles, config_updated = self._insert_initial_vehicles(
                    conn, parser, config, zone_edge_map, positive_zones
                )
                conn.commit()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(82, "Populating scheduler assignments...")
            with self.sim_db.connect() as conn:
                scheduled_count = self._populate_scheduler_from_config(conn, config)
                conn.commit()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(92, "Finalizing metadata...")
            config_hash = compute_config_hash(config)
            self.sim_db.write_preparation_fingerprint(config_hash)
            emit(100, "Simulation DB generated successfully.")

            return {
                "db_path": str(self.sim_db.db_path),
                "config_hash": config_hash,
                "network_path": str(net_path),
                "road_count": len(parser.get_edges()),
                "junction_count": len(parser.get_junctions()),
                "zone_count": len(zone_edge_map),
                "landmark_count": len(landmarks_map),
                "vehicle_count": vehicles_count,
                "scheduled_count": scheduled_count,
                "required_vehicles": required_vehicles,
                "config_updated": config_updated,
                "total_num_vehicles": int(config.get("vehicle_generation", {}).get("total_num_vehicles", 0)),
            }
        except Exception:
            if cancelled() and self.sim_db.db_path.exists():
                try:
                    self.sim_db.db_path.unlink()
                except Exception:
                    pass
            raise

    def _validate_config(self, config: Dict) -> None:
        """Validate simulation config rules required by preparation."""
        self._validate_vehicle_generation(config)
        self._validate_landmarks(config)
        self._validate_weekday_schedule(config)

    def _validate_vehicle_generation(self, config: Dict) -> None:
        vg = config.get("vehicle_generation")
        if not isinstance(vg, dict):
            raise ValueError("Missing or invalid 'vehicle_generation' section.")

        zone_allocation = vg.get("zone_allocation")
        if not isinstance(zone_allocation, dict) or not zone_allocation:
            raise ValueError("Missing or invalid 'vehicle_generation.zone_allocation'.")

        total_zone_pct = 0.0
        for zone_name, payload in zone_allocation.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Zone allocation entry '{zone_name}' must be an object.")
            pct = payload.get("percentage")
            if not isinstance(pct, (int, float)):
                raise ValueError(f"Zone '{zone_name}' percentage must be numeric.")
            total_zone_pct += float(pct)

            if str(zone_name).lower() != "noise":
                distribution = payload.get("vehicle_type_distribution")
                if not isinstance(distribution, dict) or not distribution:
                    raise ValueError(
                        f"Zone '{zone_name}' must include 'vehicle_type_distribution'."
                    )
                dist_total = 0.0
                for vt_name, vt_pct in distribution.items():
                    if not isinstance(vt_pct, (int, float)):
                        raise ValueError(
                            f"Zone '{zone_name}' vehicle type '{vt_name}' percentage must be numeric."
                        )
                    dist_total += float(vt_pct)
                if abs(dist_total - 100.0) > 0.01:
                    raise ValueError(
                        f"Zone '{zone_name}' vehicle_type_distribution must total 100 (got {dist_total:.2f})."
                    )

        if abs(total_zone_pct - 100.0) > 0.01:
            raise ValueError(
                f"Sum of all zone percentages (including noise) must be 100 (got {total_zone_pct:.2f})."
            )

        vehicle_types = vg.get("vehicle_types")
        if not isinstance(vehicle_types, dict) or not vehicle_types:
            raise ValueError("Missing or invalid 'vehicle_generation.vehicle_types'.")

    def _validate_landmarks(self, config: Dict) -> None:
        landmarks = config.get("landmarks")
        if landmarks is None:
            return
        if not isinstance(landmarks, dict):
            raise ValueError("'landmarks' must be an object when provided.")
        default_landmarks = landmarks.get("default_landmarks")
        if default_landmarks is None:
            return
        if not isinstance(default_landmarks, dict):
            raise ValueError("'landmarks.default_landmarks' must be an object.")
        visit_count = default_landmarks.get("visit_count", 1)
        if not isinstance(visit_count, int) or visit_count < 1 or visit_count > 3:
            raise ValueError("'landmarks.default_landmarks.visit_count' must be an integer in [1, 3].")

    def _validate_weekday_schedule(self, config: Dict) -> None:
        schedule = config.get("weekday_schedule", [])
        if not isinstance(schedule, list):
            raise ValueError("'weekday_schedule' must be an array.")

        for idx, item in enumerate(schedule, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"weekday_schedule entry #{idx} must be an object.")

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"weekday_schedule entry #{idx} has invalid 'name'.")

            start_time = item.get("start_time")
            end_time = item.get("end_time")
            if not isinstance(start_time, str) or not _HHMM_RE.match(start_time):
                raise ValueError(f"weekday_schedule '{name}' has invalid 'start_time'.")
            if not isinstance(end_time, str) or not _HHMM_RE.match(end_time):
                raise ValueError(f"weekday_schedule '{name}' has invalid 'end_time'.")
            # Allow overnight windows (e.g. 23:00 -> 06:00), matching legacy flow.

            repeat_on_days = item.get("repeat_on_days")
            if not isinstance(repeat_on_days, list) or not repeat_on_days:
                raise ValueError(f"weekday_schedule '{name}' must include repeat_on_days.")
            for day in repeat_on_days:
                if not isinstance(day, int) or day < 1 or day > 7:
                    raise ValueError(
                        f"weekday_schedule '{name}' repeat_on_days values must be integers in [1, 7]."
                    )

            vpm_rate = item.get("vpm_rate")
            if not isinstance(vpm_rate, (int, float)) or float(vpm_rate) <= 0 or float(vpm_rate) > 100:
                raise ValueError(
                    f"weekday_schedule '{name}' vpm_rate must be in (0, 100]."
                )

            for field_name in ("source_zones", "origin", "destination"):
                values = item.get(field_name)
                if not isinstance(values, list) or not values:
                    raise ValueError(f"weekday_schedule '{name}' must include non-empty '{field_name}'.")
                for v in values:
                    if not isinstance(v, str) or not v.strip():
                        raise ValueError(
                            f"weekday_schedule '{name}' field '{field_name}' must contain non-empty strings."
                        )

    @staticmethod
    def _hhmm_to_minutes(value: str) -> int:
        """Convert HH:MM string to minute-of-day integer."""
        hh, mm = value.split(":")
        return int(hh) * 60 + int(mm)

    def _build_zone_maps(self, parser: NetworkParser) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Build zone->edges and zone->nodes mappings using edge/node/junction zone attributes."""
        edge_zone_attr, node_zone_attr = self._load_zone_attributes_from_network_file(parser)
        zone_edges: Dict[str, List[str]] = {}
        zone_nodes: Dict[str, List[str]] = {}

        for edge_id, edge_data in parser.get_edges().items():
            zone_name = (
                edge_zone_attr.get(edge_id)
                or node_zone_attr.get(edge_data.get("from", ""))
                or node_zone_attr.get(edge_data.get("to", ""))
                or self._extract_zone_name_from_id(edge_id)
            )
            if not zone_name:
                continue
            zone_edges.setdefault(zone_name, []).append(edge_id)

        node_source = parser.get_nodes() or parser.get_junctions()
        for node_id in node_source.keys():
            zone_name = node_zone_attr.get(node_id) or self._extract_zone_name_from_id(node_id)
            if not zone_name:
                continue
            zone_nodes.setdefault(zone_name, []).append(node_id)

        for z in list(zone_edges.keys()):
            zone_edges[z] = sorted(set(zone_edges[z]))
        for z in list(zone_nodes.keys()):
            zone_nodes[z] = sorted(set(zone_nodes[z]))
        return zone_edges, zone_nodes

    def _build_landmark_map(self, config: Dict) -> Dict[str, List[str]]:
        """Return landmark mapping from config, excluding default_landmarks helper object."""
        landmarks = config.get("landmarks", {})
        if not isinstance(landmarks, dict):
            return {}
        out: Dict[str, List[str]] = {}
        for name, value in landmarks.items():
            if name == "default_landmarks":
                continue
            if isinstance(value, list):
                out[name] = [str(v) for v in value if isinstance(v, str)]
        return out

    def _get_positive_zones_from_config(self, config: Dict) -> List[str]:
        zone_alloc = config.get("vehicle_generation", {}).get("zone_allocation", {})
        if not isinstance(zone_alloc, dict):
            return []
        out = []
        for zone_name, payload in zone_alloc.items():
            if str(zone_name).lower() == "noise":
                continue
            if not isinstance(payload, dict):
                continue
            pct = float(payload.get("percentage", 0.0))
            if pct > 0:
                out.append(str(zone_name))
        return sorted(out)

    def _insert_junctions(
        self,
        conn: sqlite3.Connection,
        parser: NetworkParser,
        zone_node_map: Dict[str, List[str]],
    ) -> None:
        node_to_zone = {}
        for zone_name, node_ids in zone_node_map.items():
            for node_id in node_ids:
                node_to_zone[node_id] = zone_name
        rows = []
        for junc_id, j in parser.get_junctions().items():
            incoming = []
            outgoing = []
            if junc_id in parser.get_nodes():
                node_entry = parser.get_nodes().get(junc_id, {})
                incoming = list(node_entry.get("incoming_roads", [])) if isinstance(node_entry, dict) else []
                outgoing = list(node_entry.get("outgoing_roads", [])) if isinstance(node_entry, dict) else []
            rows.append(
                (
                    junc_id,
                    0,
                    float(j.get("x", 0.0)),
                    float(j.get("y", 0.0)),
                    str(j.get("type", "priority")),
                    node_to_zone.get(junc_id),
                    json.dumps(sorted(incoming), ensure_ascii=False),
                    json.dumps(sorted(outgoing), ensure_ascii=False),
                )
            )
        conn.executemany(
            """
            INSERT INTO junctions (id, node_type, x, y, type, zone, incoming_roads_json, outgoing_roads_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_roads(
        self,
        conn: sqlite3.Connection,
        parser: NetworkParser,
        zone_edge_map: Dict[str, List[str]],
    ) -> None:
        edge_to_zone = {}
        for zone_name, edge_ids in zone_edge_map.items():
            for edge_id in edge_ids:
                edge_to_zone[edge_id] = zone_name
        rows = []
        for edge_id, edge in parser.get_edges().items():
            lanes = edge.get("lanes", [])
            lane_count = len(lanes)
            length = parser.get_edge_length(edge_id)
            road_speed = float(lanes[0].get("speed", 13.89)) if lanes else 13.89
            rows.append(
                (
                    edge_id,
                    edge.get("from"),
                    edge.get("to"),
                    road_speed,
                    float(length),
                    int(lane_count),
                    edge_to_zone.get(edge_id),
                    json.dumps({}, ensure_ascii=False),  # vehicles_on_road
                    0.0,  # density
                    0.0,  # avg_speed
                )
            )
        conn.executemany(
            """
            INSERT INTO roads (id, from_junction, to_junction, speed, length, num_lanes, zone, vehicles_on_road_json, density, avg_speed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_zones(
        self,
        conn: sqlite3.Connection,
        config: Dict,
        zone_edge_map: Dict[str, List[str]],
        zone_node_map: Dict[str, List[str]],
    ) -> None:
        zone_alloc = config.get("vehicle_generation", {}).get("zone_allocation", {})
        rows = []
        zone_names = sorted(set(zone_edge_map.keys()) | set(zone_node_map.keys()) | set(zone_alloc.keys()))
        for zone_name in zone_names:
            payload = zone_alloc.get(zone_name, {}) if isinstance(zone_alloc, dict) else {}
            pct = float(payload.get("percentage", 0.0)) if isinstance(payload, dict) else 0.0
            rows.append(
                (
                    zone_name,
                    None,
                    pct,
                    json.dumps(sorted(zone_edge_map.get(zone_name, [])), ensure_ascii=False),
                    json.dumps(sorted(zone_node_map.get(zone_name, [])), ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),  # original_vehicles
                    json.dumps([], ensure_ascii=False),  # current_vehicles
                )
            )
        conn.executemany(
            """
            INSERT INTO zones (id, description, percentage, edges_json, junctions_json, original_vehicles_json, current_vehicles_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_landmarks(self, conn: sqlite3.Connection, landmarks_map: Dict[str, List[str]]) -> None:
        rows = []
        for name, edge_ids in sorted(landmarks_map.items()):
            rows.append((name, json.dumps(edge_ids, ensure_ascii=False), json.dumps({"name": name}, ensure_ascii=False)))
        if rows:
            conn.executemany(
                """
                INSERT INTO landmarks (landmark_name, edge_ids_json, payload_json)
                VALUES (?, ?, ?)
                """,
                rows,
            )

    def _insert_vehicle_types(self, conn: sqlite3.Connection, config: Dict) -> None:
        vehicle_types = config.get("vehicle_generation", {}).get("vehicle_types", {})
        rows = []
        for type_name, attrs in sorted(vehicle_types.items()):
            if not isinstance(attrs, dict):
                continue
            rows.append(
                (
                    type_name,
                    float(attrs.get("length", 4.5)),
                    float(attrs.get("width", 1.8)),
                    float(attrs.get("height", 1.5)),
                    float(attrs.get("max_speed", 30.0)),
                    str(attrs.get("color", "gray")),
                    json.dumps(attrs, ensure_ascii=False),
                )
            )
        conn.executemany(
            """
            INSERT INTO vehicle_types (type_name, length, width, height, max_speed, color, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_schedule_windows(self, conn: sqlite3.Connection, config: Dict, positive_zones: List[str]) -> None:
        schedule = config.get("weekday_schedule", [])
        if not isinstance(schedule, list):
            return

        default_landmarks = {}
        landmarks = config.get("landmarks", {})
        if isinstance(landmarks, dict) and isinstance(landmarks.get("default_landmarks"), dict):
            default_landmarks = landmarks.get("default_landmarks", {})
        visit_count = int(default_landmarks.get("visit_count", 1)) if isinstance(default_landmarks, dict) else 1
        visit_count = max(1, min(3, visit_count))

        rows = []
        for item in schedule:
            if not isinstance(item, dict):
                continue
            origin_expanded = self._expand_alias_landmarks(item.get("origin", []), positive_zones, visit_count)
            destination_expanded = self._expand_alias_landmarks(item.get("destination", []), positive_zones, visit_count)
            rows.append(
                (
                    str(item.get("name", "")),
                    str(item.get("start_time", "")),
                    str(item.get("end_time", "")),
                    json.dumps(item.get("repeat_on_days", []), ensure_ascii=False),
                    float(item.get("vpm_rate", 1)),
                    json.dumps(item.get("source_zones", []), ensure_ascii=False),
                    json.dumps(origin_expanded, ensure_ascii=False),
                    json.dumps(destination_expanded, ensure_ascii=False),
                    json.dumps(item, ensure_ascii=False),
                )
            )
        if rows:
            conn.executemany(
                """
                INSERT INTO schedule_windows
                (name, start_time, end_time, repeat_on_days_json, vpm_rate, source_zones_json, origin_json, destination_json, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _insert_initial_vehicles(
        self,
        conn: sqlite3.Connection,
        parser: NetworkParser,
        config: Dict,
        zone_edge_map: Dict[str, List[str]],
        positive_zones: List[str],
    ) -> Tuple[int, int, bool]:
        vg = config.get("vehicle_generation", {})
        zone_alloc = vg.get("zone_allocation", {})
        vehicle_types = vg.get("vehicle_types", {})

        dev_fraction = float(vg.get("dev_fraction", 1.0))
        original_total_vehicles = int(vg.get("total_num_vehicles", 0))
        total_vehicles = round(original_total_vehicles * dev_fraction)
        original_required = self._estimate_required_vehicles(config)
        required_vehicles = round(original_required * dev_fraction)
        config_updated = False
        if required_vehicles > total_vehicles:
            updated_total_num_vehicles = max(original_total_vehicles, int(original_required))
            vg["total_num_vehicles"] = int(updated_total_num_vehicles)
            self._write_simulation_config(config)
            total_vehicles = round(float(updated_total_num_vehicles) * dev_fraction)
            config_updated = True

        active_zone_ids = [z for z in positive_zones if z.upper() != "H"]
        if not active_zone_ids or total_vehicles <= 0:
            return 0, required_vehicles, config_updated

        noise_per_zone: Dict[str, int] = {}
        sum_vehicles = 0
        noise_payload = zone_alloc.get("noise", {}) if isinstance(zone_alloc, dict) else {}
        noise_pct = float(noise_payload.get("percentage", 0.0)) if isinstance(noise_payload, dict) else 0.0
        total_noise = round((noise_pct / 100.0) * total_vehicles)
        base_noise = total_noise // len(active_zone_ids)
        extra_noise = total_noise % len(active_zone_ids)
        for i, zid in enumerate(active_zone_ids):
            noise_per_zone[zid] = base_noise + (1 if i < extra_noise else 0)
            sum_vehicles += noise_per_zone[zid]

        zone_vehicle_counts: Dict[str, int] = {}
        for zid in active_zone_ids:
            payload = zone_alloc.get(zid, {})
            pct = float(payload.get("percentage", 0.0)) if isinstance(payload, dict) else 0.0
            num = round((pct / 100.0) * total_vehicles)
            zone_vehicle_counts[zid] = num
            sum_vehicles += num

        while sum_vehicles < total_vehicles:
            zid = random.choice(active_zone_ids)
            zone_vehicle_counts[zid] += 1
            sum_vehicles += 1

        default_landmarks = {}
        landmarks = config.get("landmarks", {})
        if isinstance(landmarks, dict) and isinstance(landmarks.get("default_landmarks"), dict):
            default_landmarks = landmarks.get("default_landmarks", {})
        home_zones = [z for z in default_landmarks.get("home_zones", []) if z in active_zone_ids] if isinstance(default_landmarks, dict) else []
        work_zones = [z for z in default_landmarks.get("work_zones", []) if z in active_zone_ids] if isinstance(default_landmarks, dict) else []
        restaurants_zones = [z for z in default_landmarks.get("restaurants_zones", []) if z in active_zone_ids] if isinstance(default_landmarks, dict) else []
        visit_count = int(default_landmarks.get("visit_count", 1)) if isinstance(default_landmarks, dict) else 1
        visit_count = max(1, min(3, visit_count))

        rows = []
        zone_original: Dict[str, List[str]] = {}
        zone_current: Dict[str, List[str]] = {}
        vehicle_id_counter = 0
        for zone_id in active_zone_ids:
            zone_cfg = zone_alloc.get(zone_id, {})
            if not isinstance(zone_cfg, dict):
                continue

            num_zone_vehicles = zone_vehicle_counts.get(zone_id, 0) + noise_per_zone.get(zone_id, 0)
            type_distribution = zone_cfg.get("vehicle_type_distribution", {})
            if not isinstance(type_distribution, dict):
                continue

            all_zone_edges = zone_edge_map.get(zone_id, [])
            eligible_roads = []
            for edge_id in all_zone_edges:
                edge = parser.get_edges().get(edge_id, {})
                lane_count = len(edge.get("lanes", []))
                if lane_count == 1:
                    eligible_roads.append(edge_id)
            if not eligible_roads:
                eligible_roads = list(all_zone_edges)
            if not eligible_roads:
                continue

            type_allocations = {
                vtype: round((float(vperc) / 100.0) * num_zone_vehicles)
                for vtype, vperc in type_distribution.items()
                if vtype in vehicle_types
            }
            vehicle_specs = [
                (vtype, vehicle_types[vtype])
                for vtype, count in type_allocations.items()
                for _ in range(max(0, int(count)))
            ]
            while len(vehicle_specs) < num_zone_vehicles and type_allocations:
                vtype = random.choice(list(type_allocations.keys()))
                vehicle_specs.append((vtype, vehicle_types[vtype]))
            random.shuffle(vehicle_specs)

            per_road = len(vehicle_specs) // len(eligible_roads)
            overflow = len(vehicle_specs) % len(eligible_roads)
            vehicle_iter = iter(vehicle_specs)
            for i, road_id in enumerate(eligible_roads):
                vehicles_on_road = per_road + (1 if i < overflow else 0)
                length = parser.get_edge_length(road_id)
                spacing = length / (vehicles_on_road + 1) if vehicles_on_road > 0 else 0.0
                edge = parser.get_edges().get(road_id, {})
                lanes = edge.get("lanes", [])
                start_x, start_y = 0.0, 0.0
                if lanes and lanes[0].get("shape"):
                    start_x, start_y = lanes[0]["shape"][0]

                for j in range(vehicles_on_road):
                    try:
                        vtype, attrs = next(vehicle_iter)
                    except StopIteration:
                        break

                    veh_id = f"veh_{vehicle_id_counter}"
                    position = (j + 1) * spacing
                    is_noise = j < noise_per_zone.get(zone_id, 0)
                    home_zone = random.choice(home_zones) if home_zones else zone_id
                    work_zone = random.choice(work_zones) if work_zones else zone_id
                    restaurant_prefs = [f"restaurant{z}" for z in restaurants_zones]
                    visit_prefs = [f"visit{k}" for k in range(1, visit_count + 1)]
                    payload = {
                        "home": {"edge": road_id, "position": position},
                        "work": None,
                        "friend1": None,
                        "friend2": None,
                        "friend3": None,
                        "park1": None,
                        "park2": None,
                        "park3": None,
                        "park4": None,
                        "stadium1": None,
                        "stadium2": None,
                        "restaurantA": None,
                        "restaurantB": None,
                        "restaurantC": None,
                    }
                    rows.append(
                        (
                            veh_id,
                            1,  # node_type
                            vtype,
                            float(attrs.get("length", 4.5)),
                            float(attrs.get("width", 1.8)),
                            float(attrs.get("height", 1.5)),
                            "white" if is_noise else str(attrs.get("color", "gray")),
                            0.0,  # speed
                            0.0,  # acceleration
                            float(start_x),
                            float(start_y),
                            zone_id,
                            road_id,
                            float(position),
                            "parked",
                            json.dumps([False, False, False, False], ensure_ascii=False),
                            1 if is_noise else 0,
                            json.dumps([], ensure_ascii=False),  # route
                            0.0,  # route_length
                            json.dumps([], ensure_ascii=False),  # route_left
                            0.0,  # route_length_left
                            None,  # origin_name
                            None,  # origin_zone
                            None,  # origin_edge
                            None,  # origin_position
                            None,  # origin_x
                            None,  # origin_y
                            None,  # origin_start_sec
                            None,  # destination_name
                            None,  # destination_zone
                            None,  # destination_edge
                            None,  # destination_position
                            None,  # destination_x
                            None,  # destination_y
                            None,  # destination_step
                            home_zone,
                            work_zone,
                            json.dumps(restaurant_prefs, ensure_ascii=False),
                            json.dumps(visit_prefs, ensure_ascii=False),
                            json.dumps(payload, ensure_ascii=False),
                        )
                    )
                    zone_original.setdefault(zone_id, []).append(veh_id)
                    zone_current.setdefault(zone_id, []).append(veh_id)
                    vehicle_id_counter += 1
                    if vehicle_id_counter >= total_vehicles:
                        break
                if vehicle_id_counter >= total_vehicles:
                    break
            if vehicle_id_counter >= total_vehicles:
                break

        if rows:
            conn.executemany(
                """
                INSERT INTO vehicles
                (
                    id, node_type, vehicle_type, length, width, height, color,
                    speed, acceleration, current_x, current_y, current_zone, current_edge, current_position,
                    status, scheduled_json, is_stagnant, route_json, route_length, route_left_json, route_length_left,
                    origin_name, origin_zone, origin_edge, origin_position, origin_x, origin_y, origin_start_sec,
                    destination_name, destination_zone, destination_edge, destination_position, destination_x, destination_y, destination_step,
                    home_zone, work_zone, restaurant_preferences_json, visit_preferences_json, destinations_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        # Keep Zone entity parity with legacy flow before scheduler population.
        for zone_id, vehicle_ids in zone_original.items():
            conn.execute(
                """
                UPDATE zones
                SET original_vehicles_json = ?, current_vehicles_json = ?
                WHERE id = ?
                """,
                (
                    json.dumps(sorted(vehicle_ids), ensure_ascii=False),
                    json.dumps(sorted(zone_current.get(zone_id, [])), ensure_ascii=False),
                    zone_id,
                ),
            )
        return len(rows), required_vehicles, config_updated

    def _estimate_required_vehicles(self, config: Dict) -> int:
        """
        Estimate required vehicles using legacy SimManager logic.

        Mirrors Traffic-DSTG-Gen estimate_required_vehicles:
        total += dispatches_per_window * num_source_zones * num_repeat_days
        """
        total = 0
        for entry in config.get("weekday_schedule", []):
            if not isinstance(entry, dict):
                continue
            vpm = float(entry.get("vpm_rate", 0.0))
            interval = 60.0 / max(vpm, 0.01)

            start = self._hhmm_to_minutes(str(entry.get("start_time", "00:00"))) * 60
            end = self._hhmm_to_minutes(str(entry.get("end_time", "00:00"))) * 60
            duration = max(end - start, 0)

            num_dispatches = int(duration // interval)
            num_zones = len(entry.get("source_zones", []))
            num_days = len(entry.get("repeat_on_days", [1, 2, 3, 4, 5]))
            total += num_dispatches * num_zones * num_days
        return int(total)

    def _write_simulation_config(self, config: Dict) -> None:
        """Persist updated simulation config JSON to project file."""
        with open(self.simulation_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _populate_scheduler_from_config(self, conn: sqlite3.Connection, config: Dict) -> int:
        """
        Populate scheduler assignments in DB, mirroring legacy schedule_from_config flow.

        Creates step-level dispatch assignments per vehicle.
        """
        schedule_entries = config.get("weekday_schedule", [])
        vg = config.get("vehicle_generation", {})
        num_weeks = int(vg.get("simulation_weeks", 1))
        dev_fraction = float(vg.get("dev_fraction", 1.0))
        seconds_in_day = 86400
        seconds_in_week = seconds_in_day * 7

        # window lookup by name for FK linkage when possible
        window_id_by_name: Dict[str, int] = {}
        for row in conn.execute("SELECT window_id, name FROM schedule_windows").fetchall():
            window_id_by_name[str(row[1])] = int(row[0])

        # zone -> vehicle pool (legacy: zone.current_vehicles)
        zone_current_vehicles: Dict[str, List[str]] = {}
        for row in conn.execute("SELECT id, current_vehicles_json FROM zones").fetchall():
            zone_id = str(row[0])
            try:
                vids = json.loads(row[1]) if row[1] else []
            except Exception:
                vids = []
            zone_current_vehicles[zone_id] = [str(v) for v in vids if isinstance(v, str)]

        # vehicle scheduled flags by week, mirrored from legacy vehicle.scheduled[week]
        vehicle_scheduled: Dict[str, List[bool]] = {}
        for row in conn.execute("SELECT id FROM vehicles").fetchall():
            vid = str(row[0])
            vehicle_scheduled[vid] = [False] * max(num_weeks, 1)

        assignment_rows = []
        total_scheduled = 0

        for week in range(num_weeks):
            week_start = week * seconds_in_week
            for entry in schedule_entries:
                if not isinstance(entry, dict):
                    continue
                entry_name = str(entry.get("name", ""))
                start_sec_local = self._hhmm_to_minutes(str(entry.get("start_time", "00:00"))) * 60
                end_sec_local = self._hhmm_to_minutes(str(entry.get("end_time", "00:00"))) * 60
                # support overnight windows
                if end_sec_local <= start_sec_local:
                    end_sec_local += seconds_in_day
                start_sec = start_sec_local + week_start
                end_sec = end_sec_local + week_start

                vpm_rate = float(entry.get("vpm_rate", 0.0)) * dev_fraction
                if vpm_rate <= 0:
                    continue
                interval = max(int(60 // vpm_rate), 1)
                local_steps = list(range(start_sec, end_sec + 1, int(interval)))
                if not local_steps:
                    local_steps = [start_sec]

                source_zones = entry.get("source_zones", [])
                origin_keys = [str(v) for v in entry.get("origin", []) if isinstance(v, str)]
                destination_keys = [str(v) for v in entry.get("destination", []) if isinstance(v, str)]
                repeat_days = [int(d) for d in entry.get("repeat_on_days", []) if isinstance(d, int)]
                if not origin_keys or not destination_keys:
                    continue

                for day in repeat_days:
                    base_step = (day - 1) * seconds_in_day
                    steps = [base_step + s for s in local_steps]
                    for zone_id in source_zones:
                        zone_id = str(zone_id)
                        eligible = [
                            vid
                            for vid in zone_current_vehicles.get(zone_id, [])
                            if vid in vehicle_scheduled and not vehicle_scheduled[vid][week]
                        ]
                        eligible = eligible[: int(len(eligible) * dev_fraction)]
                        steps_to_use = steps[: int(len(eligible))]
                        for step, veh_id in zip(steps_to_use, eligible):
                            vehicle_scheduled[veh_id][week] = True

                            # pick different origin/destination labels
                            origin = random.choice(origin_keys)
                            dest = random.choice(destination_keys)
                            attempts = 0
                            while origin == dest and attempts < 20:
                                origin = random.choice(origin_keys)
                                dest = random.choice(destination_keys)
                                attempts += 1
                            if origin == dest:
                                continue

                            payload = {
                                "entry_name": entry_name,
                                "zone_id": zone_id,
                                "week_index": week,
                                "day_of_week": day,
                                "simulation_step": int(step),
                                "origin_name": origin,
                                "destination_name": dest,
                            }
                            assignment_rows.append(
                                (
                                    veh_id,
                                    window_id_by_name.get(entry_name),
                                    int(step),
                                    int(week),
                                    int(day),
                                    origin,
                                    dest,
                                    0,
                                    json.dumps(payload, ensure_ascii=False),
                                )
                            )
                            total_scheduled += 1

        if assignment_rows:
            conn.executemany(
                """
                INSERT INTO vehicle_schedule_assignments
                (vehicle_id, window_id, simulation_step, week_index, day_of_week, origin_name, destination_name, priority, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                assignment_rows,
            )

        # Persist scheduled flags into vehicle rows.
        for vid, flags in vehicle_scheduled.items():
            conn.execute(
                "UPDATE vehicles SET scheduled_json = ? WHERE id = ?",
                (json.dumps(flags, ensure_ascii=False), vid),
            )
        return total_scheduled

    def _load_zone_attributes_from_network_file(self, parser: NetworkParser) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse explicit zone attributes from net file for edges and nodes/junctions."""
        import gzip
        import xml.etree.ElementTree as ET

        edge_zone_map: Dict[str, str] = {}
        node_zone_map: Dict[str, str] = {}
        net_path = Path(parser.net_file)
        if not net_path.exists():
            return edge_zone_map, node_zone_map

        try:
            if net_path.suffix == ".gz" or net_path.name.endswith(".gz"):
                with gzip.open(net_path, "rb") as f:
                    tree = ET.parse(f)
            else:
                tree = ET.parse(net_path)
            root = tree.getroot()
        except Exception:
            return edge_zone_map, node_zone_map

        def get_zone_value(elem) -> Optional[str]:
            zone_attr = elem.get("zone")
            if zone_attr:
                return zone_attr.strip()
            for param in elem.findall("param"):
                if (param.get("key") or "").strip().lower() == "zone":
                    value = (param.get("value") or "").strip()
                    if value:
                        return value
            return None

        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            if not edge_id:
                continue
            zone_value = get_zone_value(edge)
            if zone_value:
                edge_zone_map[edge_id] = zone_value

        for node in root.findall("node"):
            node_id = node.get("id")
            if not node_id:
                continue
            zone_value = get_zone_value(node)
            if zone_value:
                node_zone_map[node_id] = zone_value

        for junction in root.findall("junction"):
            node_id = junction.get("id")
            if not node_id:
                continue
            zone_value = get_zone_value(junction)
            if zone_value:
                node_zone_map[node_id] = zone_value

        return edge_zone_map, node_zone_map

    def _extract_zone_name_from_id(self, entity_id: str) -> Optional[str]:
        """Fallback heuristic: first alphabetic character from ID."""
        if not entity_id:
            return None
        match = re.search(r"[A-Za-z]", entity_id)
        if not match:
            return None
        return match.group(0).upper()

    def _expand_alias_landmarks(self, raw_values: Any, positive_zones: List[str], visit_count: int) -> List[str]:
        """Expand restaurants/visit aliases into explicit landmark names."""
        values = raw_values if isinstance(raw_values, list) else []
        expanded: List[str] = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            key = value.strip()
            if not key:
                continue
            if key == "restaurants":
                for zone_name in positive_zones:
                    name = f"restaurant{zone_name}"
                    if name not in seen:
                        seen.add(name)
                        expanded.append(name)
                continue
            if key == "visit":
                for idx in range(1, visit_count + 1):
                    name = f"visit{idx}"
                    if name not in seen:
                        seen.add(name)
                        expanded.append(name)
                continue
            if key not in seen:
                seen.add(key)
                expanded.append(key)
        return expanded
