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
import shutil
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

    def validate_db_readiness_for_current_config(self) -> Tuple[bool, str]:
        """
        Check whether simulation DB is ready for current simulation.config.json.

        Readiness requires:
        - DB exists
        - schema version matches
        - stored config hash matches current config hash
        """
        try:
            _config, config_hash = self.load_and_validate_config()
        except Exception as exc:
            return False, f"Simulation config invalid: {exc}"
        return self.sim_db.validate_ready_for_config(config_hash)

    def get_simulation_limit_seconds(self) -> int:
        """
        Return simulation end time in seconds (legacy calculate_simulation_limit).

        Formula: num_weeks * seconds_per_week + 1800 (30 min extra for finalization).
        If config is missing or invalid, returns 86400 + 1800 (1 day + 30 min).
        """
        seconds_in_day = 86400
        seconds_in_week = seconds_in_day * 7
        extra_time = 1800  # 30 minutes
        default = seconds_in_week + extra_time  # 1 week + 30 min if no config
        if not self.simulation_config_path.exists():
            return default
        try:
            with open(self.simulation_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            vg = config.get("vehicle_generation")
            if not isinstance(vg, dict):
                return default
            num_weeks = int(vg.get("simulation_weeks", 1))
            num_weeks = max(1, num_weeks)
            return num_weeks * seconds_in_week + extra_time
        except Exception:
            return default

    def clear_roads_and_zones(self) -> Dict[str, int]:
        """
        Clear runtime traffic state for roads/zones before simulation dispatch.

        Mirrors legacy clear_roads_and_zones behavior:
        - roads: clear vehicles, reset density/avg_speed
        - zones: clear original/current vehicles
        """
        with self.sim_db.connect() as conn:
            road_count_row = conn.execute("SELECT COUNT(*) FROM roads").fetchone()
            zone_count_row = conn.execute("SELECT COUNT(*) FROM zones").fetchone()
            road_count = int(road_count_row[0]) if road_count_row else 0
            zone_count = int(zone_count_row[0]) if zone_count_row else 0

            conn.execute(
                """
                UPDATE roads
                SET vehicles_on_road_json = ?, density = 0.0, avg_speed = 0.0
                """,
                (json.dumps([], ensure_ascii=False),),
            )
            conn.execute(
                """
                UPDATE zones
                SET original_vehicles_json = ?, current_vehicles_json = ?
                """,
                (json.dumps([], ensure_ascii=False), json.dumps([], ensure_ascii=False)),
            )
            conn.commit()

        return {"roads_cleared": road_count, "zones_cleared": zone_count}

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

            emit(22, "Resetting mapping directory...")
            self.delete_mapping_dir_if_exists()
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

            emit(76, "Assigning destinations (origin/destination labels → edges)...")
            with self.sim_db.connect() as conn:
                self._assign_destinations_from_config(conn, parser, config, zone_edge_map, landmarks_map, positive_zones)
                conn.commit()
            if cancelled():
                raise RuntimeError("Preparation cancelled.")

            emit(82, "Populating scheduler assignments (step, origin, destination per vehicle)...")
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

        # Same zone→edge logic as GUI "Zones" section Show button (detect_zones_from_network).
        for edge_id, edge_data in parser.get_edges().items():
            zone_name = edge_zone_attr.get(edge_id)
            if not zone_name:
                zone_name = node_zone_attr.get(edge_data.get("from", ""))
            if not zone_name:
                zone_name = node_zone_attr.get(edge_data.get("to", ""))
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_id)
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_data.get("from", ""))
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_data.get("to", ""))
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
            noise_assigned_in_zone = 0  # first N vehicles in this zone are noise
            type_distribution = zone_cfg.get("vehicle_type_distribution", {})
            if not isinstance(type_distribution, dict):
                continue

            # Use all zone edges for vehicle placement (same zone→edges as GUI Show button; no single-lane filter).
            eligible_roads = list(zone_edge_map.get(zone_id, []))
            if not eligible_roads:
                continue
            # Do not shuffle: match legacy — vehicles distributed in order across zone roads

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

            # Legacy (entities.py): per_road = specs // len(eligible_roads), overflow; distribute evenly across zone roads
            per_road = len(vehicle_specs) // len(eligible_roads)
            overflow = len(vehicle_specs) % len(eligible_roads)
            vehicle_iter = iter(vehicle_specs)
            for i, road_id in enumerate(eligible_roads):
                vehicles_on_road = per_road + (1 if i < overflow else 0)
                length = parser.get_edge_length(road_id)
                # Legacy: spacing = length / (vehicles_on_road + 1); pos = (j+1)*spacing → evenly spaced along road
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
                    # Home position: evenly spaced along road (legacy: pos = (j+1)*spacing, spacing = length/(vehicles_on_road+1))
                    position = (j + 1) * spacing
                    zone_noise_limit = noise_per_zone.get(zone_id, 0)
                    is_noise = noise_assigned_in_zone < zone_noise_limit
                    if is_noise:
                        noise_assigned_in_zone += 1
                    # For debugging and consistency with GUI, use the zone itself as home_zone.
                    home_zone = zone_id
                    work_zone = random.choice(work_zones) if work_zones else zone_id
                    restaurant_prefs = [f"restaurant{z}" for z in restaurants_zones]
                    visit_prefs = [f"visit{k}" for k in range(1, visit_count + 1)]
                    work_dest = None
                    if work_zones:
                        work_edges = [
                            eid for eid in zone_edge_map.get(work_zone, [])
                            if len(parser.get_edges().get(eid, {}).get("lanes", [])) == 1
                        ]
                        if not work_edges:
                            work_edges = list(zone_edge_map.get(work_zone, []))
                        if work_edges:
                            w_road = random.choice(work_edges)
                            w_len = parser.get_edge_length(w_road)
                            w_pos = random.uniform(0.0, max(0.0, w_len - 1.0)) if w_len else 0.0
                            work_dest = {"edge": w_road, "position": w_pos}
                    # Origin for this vehicle = "home" (starting edge/position); filled in _assign_destinations_from_config
                    payload = {
                        "home": {"edge": road_id, "position": position},
                        "work": work_dest,
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
                    status, scheduled_json, is_noise, route_json, route_length, route_left_json, route_length_left,
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

    def _resolve_mapping_dir(self, dataset_output_folder: Optional[str] = None) -> Path:
        """Resolve <dataset_output>/mapping directory path."""
        if dataset_output_folder:
            output_dir = Path(dataset_output_folder).resolve()
        else:
            output = self.config_manager.get_dataset_output_folder()
            if output:
                output_dir = Path(output).resolve()
            else:
                output_dir = (self.project_path / "datasets").resolve()
        return output_dir / "mapping"

    def delete_mapping_dir_if_exists(self, dataset_output_folder: Optional[str] = None) -> bool:
        """
        Delete mapping directory if present.

        This is called on DB generation so mapping files always match the new snapshot.
        """
        mapping_dir = self._resolve_mapping_dir(dataset_output_folder)
        if mapping_dir.exists():
            if mapping_dir.is_dir():
                shutil.rmtree(mapping_dir, ignore_errors=False)
            else:
                mapping_dir.unlink()
            return True
        return False

    def create_mapping_files_if_missing(
        self,
        dataset_output_folder: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        progress_counts_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Create mapping files only when mapping directory does not exist.

        Files:
        - vehicle_mapping.json
        - junction_mapping.json
        - edge_mapping.json
        """
        def emit(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        def emit_counts(current: int, total: int) -> None:
            if progress_counts_cb:
                progress_counts_cb(int(current), int(total))

        mapping_dir = self._resolve_mapping_dir(dataset_output_folder)
        emit(f"Mapping: target directory {mapping_dir}")
        if mapping_dir.exists():
            emit("Mapping: directory already exists, skipping creation.")
            return {"created": False, "mapping_dir": str(mapping_dir)}

        emit("Mapping: creating directory...")
        mapping_dir.mkdir(parents=True, exist_ok=True)

        def natural_sort_key(text: str):
            def as_num_or_text(chunk: str):
                return int(chunk) if chunk.isdigit() else chunk.lower()
            return [as_num_or_text(c) for c in re.split(r"(\d+)", str(text))]

        def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
            obj = {}
            for key in row.keys():
                value = row[key]
                if key.endswith("_json") and isinstance(value, str):
                    try:
                        obj[key] = json.loads(value)
                        continue
                    except Exception:
                        pass
                obj[key] = value
            return obj

        emit("Mapping: reading vehicles/junctions/roads from DB...")
        with self.sim_db.connect() as conn:
            conn.row_factory = sqlite3.Row
            vehicle_rows = conn.execute("SELECT * FROM vehicles").fetchall()
            junction_rows = conn.execute("SELECT * FROM junctions").fetchall()
            road_rows = conn.execute("SELECT * FROM roads").fetchall()

        total_entities = len(vehicle_rows) + len(junction_rows) + len(road_rows)
        done = 0
        emit_counts(done, max(1, total_entities))

        emit("Mapping: building sorted mapping objects...")
        vehicle_mapping = {}
        for r in vehicle_rows:
            vehicle_mapping[str(r["id"])] = row_to_dict(r)
            done += 1
            emit_counts(done, max(1, total_entities))
        junction_mapping = {}
        for r in junction_rows:
            junction_mapping[str(r["id"])] = row_to_dict(r)
            done += 1
            emit_counts(done, max(1, total_entities))
        edge_mapping = {}
        for r in road_rows:
            edge_mapping[str(r["id"])] = row_to_dict(r)
            done += 1
            emit_counts(done, max(1, total_entities))

        vehicle_mapping = dict(sorted(vehicle_mapping.items(), key=lambda kv: natural_sort_key(kv[0])))
        junction_mapping = dict(sorted(junction_mapping.items(), key=lambda kv: natural_sort_key(kv[0])))
        edge_mapping = dict(sorted(edge_mapping.items(), key=lambda kv: natural_sort_key(kv[0])))

        emit("Mapping: writing vehicle_mapping.json...")
        with open(mapping_dir / "vehicle_mapping.json", "w", encoding="utf-8") as f:
            json.dump(vehicle_mapping, f, indent=2, ensure_ascii=False)
        emit("Mapping: writing junction_mapping.json...")
        with open(mapping_dir / "junction_mapping.json", "w", encoding="utf-8") as f:
            json.dump(junction_mapping, f, indent=2, ensure_ascii=False)
        emit("Mapping: writing edge_mapping.json...")
        with open(mapping_dir / "edge_mapping.json", "w", encoding="utf-8") as f:
            json.dump(edge_mapping, f, indent=2, ensure_ascii=False)

        emit("Mapping: completed successfully.")
        return {
            "created": True,
            "mapping_dir": str(mapping_dir),
            "vehicle_count": len(vehicle_mapping),
            "junction_count": len(junction_mapping),
            "edge_count": len(edge_mapping),
        }

    def _assign_destinations_from_config(
        self,
        conn: sqlite3.Connection,
        parser: NetworkParser,
        config: Dict,
        zone_edge_map: Dict[str, List[str]],
        landmarks_map: Dict[str, List[str]],
        active_zone_ids: List[str],
    ) -> None:
        """
        Assign origin and destinations per vehicle (legacy assign_destinations).

        - Origin: each vehicle's origin (starting location) is "home" = (current_edge, current_position).
          We set it here from the DB so destinations_json["home"] is the canonical origin for dispatch.
        - Destinations: work, friend1–3, park1–4, stadium1–2, restaurantA/B/C, visit1–3 from zones/landmarks.

        Called after _insert_initial_vehicles so dispatch can resolve origin_name/destination_name
        from vehicle_schedule_assignments to edge/position via destinations_json.
        """
        default_landmarks = {}
        landmarks = config.get("landmarks", {})
        if isinstance(landmarks, dict) and isinstance(landmarks.get("default_landmarks"), dict):
            default_landmarks = landmarks.get("default_landmarks", {})
        visit_count = max(1, min(3, int(default_landmarks.get("visit_count", 1))))
        restaurants_zones = [z for z in default_landmarks.get("restaurants_zones", []) if z in active_zone_ids] if isinstance(default_landmarks, dict) else []

        rows = conn.execute(
            "SELECT id, current_zone, current_edge, current_position, destinations_json, home_zone, work_zone "
            "FROM vehicles"
        ).fetchall()
        for row in rows:
            veh_id = str(row[0])
            current_zone = str(row[1]) if row[1] else None
            current_edge = str(row[2]) if row[2] else ""
            current_position = float(row[3] or 0.0)
            try:
                dest = json.loads(row[4]) if row[4] else {}
            except Exception:
                dest = {}
            if not isinstance(dest, dict):
                dest = {}
            home_zone = str(row[5]) if row[5] else current_zone
            work_zone = str(row[6]) if row[6] else current_zone

            # Origin: each vehicle's starting location = "home" (canonical for trip origin at dispatch)
            dest["home"] = {"edge": current_edge, "position": current_position}

            # WORK (random edge in work_zone; do not restrict to single-lane edges)
            work_edges = list(zone_edge_map.get(work_zone, []))
            if work_edges:
                w_edge = random.choice(work_edges)
                w_len = parser.get_edge_length(w_edge)
                dest["work"] = {"edge": w_edge, "position": random.uniform(1.0, max(1.0, w_len - 1.0))}

            # FRIEND1: same zone (all edges)
            same_edges = list(zone_edge_map.get(current_zone, [])) if current_zone else []
            if same_edges:
                e = random.choice(same_edges)
                ln = parser.get_edge_length(e)
                dest["friend1"] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}

            # FRIEND2, FRIEND3: other zones (all edges)
            other_zones = [z for z in active_zone_ids if z != current_zone][:2]
            for i, z in enumerate(other_zones):
                label = f"friend{i + 2}"
                other_edges = list(zone_edge_map.get(z, []))
                if other_edges:
                    e = random.choice(other_edges)
                    ln = parser.get_edge_length(e)
                    dest[label] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}

            # PARKS, STADIUMS, RESTAURANTS, VISIT from landmarks_map
            for label, edge_ids in landmarks_map.items():
                if not edge_ids:
                    continue
                e = random.choice(edge_ids)
                ln = parser.get_edge_length(e)
                dest[label] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}

            # Restaurant labels by zone if not in landmarks_map (all edges)
            for z in restaurants_zones:
                label = f"restaurant{z}"
                if label in dest:
                    continue
                rest_edges = list(zone_edge_map.get(z, []))
                if rest_edges:
                    e = random.choice(rest_edges)
                    ln = parser.get_edge_length(e)
                    dest[label] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}

            # Visit labels (visit1, visit2, visit3)
            for k in range(1, visit_count + 1):
                label = f"visit{k}"
                if label in dest:
                    continue
                if label in landmarks_map and landmarks_map[label]:
                    e = random.choice(landmarks_map[label])
                    ln = parser.get_edge_length(e)
                    dest[label] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}
                else:
                    z = random.choice(active_zone_ids) if active_zone_ids else None
                    if z:
                        visit_edges = list(zone_edge_map.get(z, []))
                        if visit_edges:
                            e = random.choice(visit_edges)
                            ln = parser.get_edge_length(e)
                            dest[label] = {"edge": e, "position": random.uniform(1.0, max(1.0, ln - 1.0))}

            conn.execute("UPDATE vehicles SET destinations_json = ? WHERE id = ?", (json.dumps(dest, ensure_ascii=False), veh_id))

    def _populate_scheduler_from_config(self, conn: sqlite3.Connection, config: Dict) -> int:
        """
        Equivalent of legacy schedule_from_config (Traffic-DSTG-Gen/graph/entities.py).

        For each week/schedule entry/day/source_zone, pairs simulation steps with eligible
        vehicles and assigns (simulation_step, origin_name, destination_name) per vehicle.
        Results are stored in vehicle_schedule_assignments. At dispatch time, the runner
        resolves origin_name/destination_name to edge/position via each vehicle's destinations_json
        (filled by _assign_destinations_from_config).
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
                    # print(f"[WARNING] scheduler no origin/destination keys for {entry_name}")
                    continue

                # print(f"[DEBUG] scheduler origin keys: {origin_keys}")
                # print(f"[DEBUG] scheduler destination keys: {destination_keys}")
                # print(f"[DEBUG] scheduler source zones: {source_zones}")

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
                        # Randomize order so that the fixed creation order (and thus home edge order)
                        # is decoupled from the chronological dispatch order.
                        random.shuffle(eligible)
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
                            # print(
                            #     f"[DEBUG] scheduler origin for {veh_id}: week={week}, day={day}, step={int(step)}, "
                            #     f"origin_name={origin}, destination_name={dest}"
                            # )
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
