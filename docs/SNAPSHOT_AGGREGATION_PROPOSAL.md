# Proposal: CSV → Step (Snapshot) Files — Direct Pipeline

## Context

- **Input**: Train CSV (Porto-style) with polyline and TIMESTAMP per row. **No intermediate traj JSONs**.
- **Output**: `step_*.json` files (one every 30 seconds) with traffic state: nodes (junctions + vehicles), edges (roads + dynamic edges).
- **Reference**: Example step file at `/media/guy/StorageVolume/traffic_data_2days/step_047970.json`.

---

## Design Overview

**Single-pass pipeline**: Sort the train CSV chronologically, then process each row directly: convert polyline → SUMO route (same logic as `convert_trajectory`), and feed the result into the chronological processing map. This avoids the 2-step operation (CSV → traj JSON → step JSON).

The aggregation is **chronological and stateful**: we process CSV rows in time order, maintain a live traffic database, and emit a step snapshot whenever we cross a 30-second boundary.

---

## Incremental Implementation Plan

### Step 1: Sort Train CSV Chronologically

**Goal**: Establish chronological order for processing.

**Actions**:
1. Load train CSV (same format as `iter_trajectories_from_csv`).
2. Parse each row to get `(trip_num, polyline, timestamp)`.
3. Sort rows by `timestamp` ascending (primary key). Secondary sort by `trip_num` if timestamps tie.
4. Yield or iterate in sorted order.

**Output**: `load_and_sort_csv(csv_path: Path, start_traj, last_traj) -> Iterator[Tuple[int, List, int]]` or sorted list.

**Deliverable**: Sorted iterator/list; earliest timestamp first. No traj JSON files created.

---

### Step 2: Static Junctions Map

**Goal**: Build a map of all static nodes (junctions) from the network.

**Actions**:
1. Load SUMO network via **sumolib** (or traci if simulation is running).
2. Iterate all junctions/nodes.
3. For each junction ID, extract features: coordinates, shape, type, incoming/outgoing edges, connections, etc.
4. Store: `junctions: Dict[str, JunctionFeatures]` where key = junction ID, value = all features.

**Data source**: `sumolib.net.readNet()` → `net.getNodes()` or `net.getNode(id)`.

**Output**: `build_junctions_map(network_path: str) -> Dict[str, Dict]`

**Deliverable**: Map of junction ID → junction features (coord, shape, type, etc.).

---

### Step 3: Roads Map (Edges) — Constantly Updated

**Goal**: Build a map of all roads (edges) where key = edge_id, value = edge features. This map is **updated continuously** as vehicles move (e.g. density, avg_speed, vehicles_on_road).

**Actions**:
1. Load network via **sumolib**.
2. Iterate all edges (exclude internal/junction edges if desired).
3. For each edge ID, extract static features: from, to, length, speed, num_lanes, shape, etc.
4. Initialize dynamic fields: `density`, `avg_speed`, `vehicles_on_road` (empty initially).
5. Store: `roads: Dict[str, EdgeFeatures]` — **this map is updated** as we process trajectories (add/remove vehicles, recompute density and avg_speed).

**Output**: `build_roads_map(network_path: str) -> Dict[str, Dict]`

**Deliverable**: Map of edge_id → edge features; dynamic fields updated during processing.

---

### Step 4: Vehicles-in-Motion Map

**Goal**: Track all vehicles currently active (on the network).

**Actions**:
1. Create map: `vehicles: Dict[str, VehicleFeatures]` where key = vehicle_id (use `trajectory_id` → `veh_{trajectory_id}`).
2. Value = all vehicle features: current position (x, y), current_edge, speed, accumulated_distance, route, route_left, etc.
3. When a trajectory **starts** (first point processed): add vehicle to map.
4. When a trajectory **ends** (last point processed, or vehicle leaves network): remove vehicle from map.
5. While processing: update vehicle state at each timestamp.

**Output**: In-memory map updated during chronological processing.

**Deliverable**: Map of vehicle_id → vehicle features; add/update/remove as we process trajectories.

---

### Step 5: Dynamic Edges (Junction → Vehicle → Vehicle → Edges)

**Goal**: Represent dynamic connectivity: junction–vehicle and vehicle–vehicle relationships, plus vehicle–edge occupancy.

**Actions**:
1. Define structure for dynamic edges (to be refined):
   - **Junction → Vehicle**: which vehicles are "at" or "near" a junction.
   - **Vehicle → Vehicle**: spatial proximity or leader–follower relationships.
   - **Vehicle → Edges**: which edges each vehicle has in its planned route (updates `roads` map).
2. Update these as vehicles move; used to enrich the traffic DB and step output.

**Output**: Structures to capture junction–vehicle, vehicle–vehicle, and vehicle–edge links.

**Deliverable**: Dynamic edge structures integrated into the traffic DB.

---

### Step 6: Traffic Database

**Goal**: Central store for all maps used during aggregation.

**Actions**:
1. Create a `TrafficDB` (or similar) class/struct that holds:
   - `junctions: Dict[str, JunctionFeatures]` (static)
   - `roads: Dict[str, EdgeFeatures]` (static base + dynamic updates)
   - `vehicles: Dict[str, VehicleFeatures]` (dynamic)
   - `dynamic_edges` (junction–vehicle, vehicle–vehicle, etc.)
2. Provide methods: `add_vehicle`, `update_vehicle`, `remove_vehicle`, `update_road_stats`, etc.
3. Single source of truth for the current traffic state at any moment.

**Output**: `TrafficDB` class in `src/utils/traffic_db.py`

**Deliverable**: DB that stores and updates all maps; used by step emission logic.

---

### Step 7: Chronological Processing — CSV Row → SUMO Route → Add to Map

**Goal**: Process each CSV row in sorted order. Convert polyline to SUMO route **in-memory** (same as `convert_trajectory`), then add data directly to the chronological processing map. **No traj JSON files.**

**Actions**:
1. Iterate sorted CSV rows chronologically (from Step 1).
2. For each row `(trip_num, polyline, timestamp)`:
   - Call `convert_trajectory(trip_num, polyline, timestamp, ...)` — same logic as trajectory conversion, returns in-memory dict with `segments` and `sumo_route_gps`.
   - If conversion succeeds: iterate `sumo_route_gps` points (flatten across segments) in time order.
   - At each timestamp T, update:
     - `vehicles[veh_id]` with: coordinates, current_edge_id, speed, accumulated_distance_traveled.
     - `roads[edge_id]` (add vehicle to edge, update density/avg_speed).
     - `dynamic_edges` (junction–vehicle, vehicle–vehicle links).
3. Build auxiliary map: `timestamp_to_locations: Dict[int, List[VehicleLocation]]` where each entry has: coordinates, current_edge_id, speed, accumulated_distance.
4. This map feeds into step emission and DB updates.

**Output**: Processing loop that: CSV row → convert_trajectory → update TrafficDB and `timestamp_to_locations`.

**Deliverable**: Single-pass pipeline; CSV → step JSONs with no intermediate traj files.

---

### Step 8: Emit Step JSON Every 30 Seconds

**Goal**: When the next trajectory (CSV row) would push us past a 30-second boundary, emit a step snapshot **before** processing that row.

**Logic**:
1. **Base timestamp** = first timestamp in the sorted dataset (or configurable).
2. **Step boundaries** = 0, 30, 60, 90, … seconds from base.
3. **Current time** = timestamp of the last processed point.
4. **Next trajectory** = next CSV row; `next_traj_timestamp` = its timestamp.
5. **Rule**: If `next_traj_timestamp > current_step_boundary` (e.g. 30), then:
   - **First**: Capture snapshot of current TrafficDB → write `step_0000030.json`.
   - **Then**: Process the next CSV row (convert → add to map).

**Example**:
- Just processed trajectory with last point at timestamp 29s → DB has active vehicles, updated roads, junctions, dynamic edges.
- Next CSV row has timestamp 35s.
- Since 35 > 30, we **first** emit `step_0000030.json` (snapshot at 30s).
- **Then** process the CSV row (convert to route, add to map).

**Output**: `emit_step_if_boundary_crossed(current_time, next_traj_time, step_interval, traffic_db, output_dir) -> Optional[int]` (returns step number if emitted).

**Deliverable**: Step files written at 30s intervals; snapshot reflects state **before** the next trajectory joins.

---

## Processing Flow (Summary)

```
1. Sort train CSV chronologically (by timestamp)
2. Build static junctions map (sumolib)
3. Build roads map with dynamic fields (sumolib)
4. Initialize vehicles map (empty)
5. Initialize dynamic edges structures
6. Create TrafficDB with all maps
7. For each CSV row (in order):
   a. If next row's timestamp would cross 30s boundary → emit step JSON first
   b. Convert row (polyline) → SUMO route via convert_trajectory (in-memory, no JSON)
   c. For each sumo_route_gps point in the result:
      - Update vehicle in vehicles map
      - Update roads map (add vehicle to edge, density, avg_speed)
      - Update dynamic edges
   d. When trajectory ends → remove vehicle from vehicles map
8. After last row → emit final step if needed
```

---

## File Layout

```
src/utils/traffic_db.py           # TrafficDB: junctions, roads, vehicles, dynamic edges
src/utils/snapshot_aggregator.py # Build maps, chronological processing, step emission
scripts/aggregate_to_steps.py     # CLI: --csv, --network, --output (no traj dir)
```

**Input**: Train CSV path (not traj JSON directory).

---

## Data Structures (Draft)

| Map | Key | Value (features) |
|-----|-----|------------------|
| Junctions | junction_id | coord, shape, type, incoming, outgoing, connections |
| Roads | edge_id | from, to, length, speed, num_lanes, density, avg_speed, vehicles_on_road |
| Vehicles | veh_{trajectory_id} | x, y, current_edge, speed, accumulated_distance, route, route_left |
| Dynamic edges | TBD | junction↔vehicle, vehicle↔vehicle links |

---

## Open Questions

1. **sumolib vs NetworkParser**: Use sumolib for junction/edge features (as specified), or align with existing `NetworkParser`? sumolib is standard for SUMO; NetworkParser has custom GPS↔SUMO conversion.
2. **Dynamic edges schema**: Exact structure for junction→vehicle, vehicle→vehicle — to be defined in Step 5.
3. **Step file naming**: `step_0000030.json` (zero-padded to 6 digits) vs `step_30.json`.
4. **Interpolation**: When a step boundary (e.g. 30s) falls between two trajectory points, interpolate or use nearest? Proposal: interpolate for accuracy.
