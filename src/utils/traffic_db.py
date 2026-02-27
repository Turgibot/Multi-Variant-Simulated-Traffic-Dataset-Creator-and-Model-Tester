"""
TrafficDB: Central store for traffic state during aggregation.

Contains:
- junctions: static junction nodes
- road_edges: edges with edge_type=0 only (static base + dynamic updates)
- vehicles: active vehicle nodes (dynamic)

Dynamic edges (edge_type 1,2,3) are NOT stored; they are built at snapshot creation time.
"""

from typing import Any, Dict, List


class TrafficDB:
    """
    Traffic database for chronological aggregation.
    """

    def __init__(
        self,
        junctions: Dict[str, Dict[str, Any]],
        road_edges: Dict[str, Dict[str, Any]],
    ):
        """
        Initialize with static junctions and road edges.
        vehicles starts empty.
        """
        self.junctions = junctions
        self.road_edges = road_edges
        self.vehicles: Dict[str, Dict[str, Any]] = {}

    def add_vehicle(self, vehicle_id: str, vehicle_data: Dict[str, Any]) -> None:
        """Add an active vehicle."""
        self.vehicles[vehicle_id] = vehicle_data

    def update_vehicle(self, vehicle_id: str, updates: Dict[str, Any]) -> None:
        """Update vehicle fields. Vehicle must exist."""
        if vehicle_id not in self.vehicles:
            return
        self.vehicles[vehicle_id].update(updates)

    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove vehicle from DB and from road_edges.vehicles_on_road."""
        if vehicle_id not in self.vehicles:
            return
        vehicle = self.vehicles.pop(vehicle_id)
        edge_id = vehicle.get("current_edge", "")
        if edge_id and edge_id in self.road_edges:
            vo = self.road_edges[edge_id].get("vehicles_on_road", [])
            if vehicle_id in vo:
                vo.remove(vehicle_id)
            self._update_edge_stats(edge_id)

    def add_vehicle_to_edge(self, vehicle_id: str, edge_id: str) -> None:
        """
        Add vehicle to an edge's vehicles_on_road and update edge stats.
        Call when vehicle enters or moves on an edge.
        Speed is read from db.vehicles when computing avg_speed.
        """
        if edge_id not in self.road_edges:
            return
        vo = self.road_edges[edge_id].setdefault("vehicles_on_road", [])
        if vehicle_id not in vo:
            vo.append(vehicle_id)
        self._update_edge_stats(edge_id)

    def remove_vehicle_from_edge(self, vehicle_id: str, edge_id: str) -> None:
        """Remove vehicle from edge's vehicles_on_road and update stats."""
        if edge_id not in self.road_edges:
            return
        vo = self.road_edges[edge_id].get("vehicles_on_road", [])
        if vehicle_id in vo:
            vo.remove(vehicle_id)
        self._update_edge_stats(edge_id)

    def _update_edge_stats(self, edge_id: str) -> None:
        """Recompute density, avg_speed for an edge from its vehicles_on_road. edge_demand TBD."""
        edge = self.road_edges.get(edge_id)
        if not edge:
            return
        vo = edge.get("vehicles_on_road", [])
        n = len(vo)
        if n == 0:
            edge["density"] = 0.0
            edge["avg_speed"] = 0.0
            # edge_demand updated elsewhere (TBD)
            return
        length = edge.get("length", 1.0)
        num_lanes = edge.get("num_lanes", 1)
        edge["density"] = n / (length * num_lanes) if length * num_lanes > 0 else 0.0
        speeds = [
            self.vehicles[vid].get("speed", 0.0)
            for vid in vo
            if vid in self.vehicles
        ]
        edge["avg_speed"] = sum(speeds) / len(speeds) if speeds else 0.0

    def update_road_stats(self, edge_id: str) -> None:
        """Public API to recompute density and avg_speed for an edge."""
        self._update_edge_stats(edge_id)
