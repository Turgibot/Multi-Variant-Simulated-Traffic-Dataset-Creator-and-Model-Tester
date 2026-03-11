"""
SQLite schema and metadata utilities for simulation preparation.

This module defines the preparation database contract only.
It does not run preparation or simulation loops.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple


SCHEMA_VERSION = 1


def compute_sha256_for_bytes(data: bytes) -> str:
    """Return SHA256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_for_file(file_path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config: Any) -> str:
    """
    Return SHA256 hash for config object using canonical JSON encoding.

    This keeps hash stable across whitespace and key-order differences.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return compute_sha256_for_bytes(payload.encode("utf-8"))


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class SimulationDB:
    """Handles DB path, schema initialization, and metadata operations."""

    def __init__(self, project_path: str, project_name: str):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.simulation_dir = self.project_path / "simulation"
        self.db_path = self.simulation_dir / f"{project_name}.db"

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection, creating parent directory if needed."""
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize_schema(self, overwrite: bool = True) -> None:
        """
        Create schema for the simulation DB.

        If overwrite is True, existing DB file is removed first.
        """
        if overwrite and self.db_path.exists():
            self.db_path.unlink()

        with self.connect() as conn:
            self._create_schema(conn)
            self.set_metadata(conn, "schema_version", str(SCHEMA_VERSION))
            self.set_metadata(conn, "project_name", self.project_name)
            self.set_metadata(conn, "db_created_at_utc", now_utc_iso())
            conn.commit()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create all schema tables for full entity snapshot + scheduler."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS junctions (
                junction_id TEXT PRIMARY KEY,
                x REAL,
                y REAL,
                zone_name TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS roads (
                edge_id TEXT PRIMARY KEY,
                from_junction_id TEXT,
                to_junction_id TEXT,
                length REAL,
                lane_count INTEGER,
                zone_name TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS zones (
                zone_name TEXT PRIMARY KEY,
                percentage REAL,
                edge_ids_json TEXT NOT NULL,
                node_ids_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS landmarks (
                landmark_name TEXT PRIMARY KEY,
                edge_ids_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vehicle_types (
                type_name TEXT PRIMARY KEY,
                length REAL,
                width REAL,
                height REAL,
                max_speed REAL,
                color TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id TEXT PRIMARY KEY,
                vehicle_type TEXT NOT NULL,
                home_zone TEXT,
                work_zone TEXT,
                restaurant_preferences_json TEXT NOT NULL,
                visit_preferences_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(vehicle_type) REFERENCES vehicle_types(type_name)
            );

            CREATE TABLE IF NOT EXISTS schedule_windows (
                window_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                repeat_on_days_json TEXT NOT NULL,
                vpm_rate INTEGER NOT NULL,
                source_zones_json TEXT NOT NULL,
                origin_json TEXT NOT NULL,
                destination_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vehicle_schedule_assignments (
                assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT NOT NULL,
                window_id INTEGER NOT NULL,
                priority INTEGER DEFAULT 0,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id),
                FOREIGN KEY(window_id) REFERENCES schedule_windows(window_id)
            );

            CREATE TABLE IF NOT EXISTS route_candidates (
                route_candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT NOT NULL,
                origin_key TEXT NOT NULL,
                destination_key TEXT NOT NULL,
                edge_sequence_json TEXT NOT NULL,
                cost REAL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id)
            );

            CREATE INDEX IF NOT EXISTS idx_roads_zone_name
                ON roads(zone_name);
            CREATE INDEX IF NOT EXISTS idx_vehicles_type
                ON vehicles(vehicle_type);
            CREATE INDEX IF NOT EXISTS idx_assignments_vehicle
                ON vehicle_schedule_assignments(vehicle_id);
            CREATE INDEX IF NOT EXISTS idx_assignments_window
                ON vehicle_schedule_assignments(window_id);
            CREATE INDEX IF NOT EXISTS idx_route_candidates_vehicle
                ON route_candidates(vehicle_id);
            """
        )

    def set_metadata(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        """Set or replace one metadata key/value pair."""
        conn.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def get_metadata(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        """Read one metadata value by key."""
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def write_preparation_fingerprint(self, config_hash: str) -> None:
        """Store preparation fingerprint metadata used for run-time validation."""
        with self.connect() as conn:
            self.set_metadata(conn, "schema_version", str(SCHEMA_VERSION))
            self.set_metadata(conn, "config_hash", config_hash)
            self.set_metadata(conn, "prepared_at_utc", now_utc_iso())
            conn.commit()

    def validate_ready_for_config(self, config_hash: str) -> Tuple[bool, str]:
        """
        Validate DB readiness against schema version and config hash.

        Returns:
            (is_valid, reason)
        """
        if not self.db_path.exists():
            return False, "Simulation DB does not exist."

        try:
            with self.connect() as conn:
                schema_version = self.get_metadata(conn, "schema_version")
                if schema_version != str(SCHEMA_VERSION):
                    return False, "Simulation DB schema version mismatch."
                stored_hash = self.get_metadata(conn, "config_hash")
                if not stored_hash:
                    return False, "Simulation DB has no config hash metadata."
                if stored_hash != config_hash:
                    return False, "Simulation DB is stale (config hash mismatch)."
                return True, "Simulation DB is ready."
        except Exception as exc:
            return False, f"Simulation DB validation failed: {exc}"
