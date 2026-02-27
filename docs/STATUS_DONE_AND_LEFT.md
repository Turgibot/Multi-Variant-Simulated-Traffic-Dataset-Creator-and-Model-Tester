# CSV → Step Pipeline: Done and Left

## Done

### Core pipeline
- [x] **Step 1**: Sort train CSV chronologically (`load_and_sort_csv`, `--sorted`, `--save-sorted`)
- [x] **Step 2**: Static junctions map (`build_junctions_map`)
- [x] **Step 3**: Roads map with dynamic fields (`build_edges_map`, `density`, `avg_speed`, `edge_demand`, `vehicles_on_road`)
- [x] **Step 4**: Vehicle tracking with `veh_<trajectory_id>_<segment_index>`
- [x] **Step 5**: Dynamic edges built at snapshot time (`create_dynamic_edges(road_edges, vehicles)`)
- [x] **Step 6**: TrafficDB (`src/utils/traffic_db.py`

### Processing flow
- [x] **timestamp_to_vehicles**: SortedDict (or dict fallback), key=ts, value=list of VehicleInfo
- [x] **Segment processing**: Add vehicle to DB (static only), add VehicleInfo per point to timestamp_to_vehicles
- [x] **Flush loop**: Pop and apply until boundary; interleaved with emit
- [x] **Emit loop**: Create JSON at each boundary after flushing
- [x] **Vehicle removal**: Remove vehicles whose trajectory ended before current trajectory starts

### Vehicle data
- [x] **Static info** (origin, destination, route) stored in DB when segment is added
- [x] **VehicleInfo** (id, speed, acceleration, current_x, current_y, current_zone, current_edge, current_position, route_left, route_length_left) stored per point in timestamp_to_vehicles
- [x] **current_position**: from projecting vehicle coords onto edge shape
- [x] **route_length_left**: remaining distance along route

### CLI
- [x] `--csv`, `--network`, `--output`, `--start`, `--last`, `--sorted`, `--save-sorted`, `--sampling-period`, `--sumo-home`
- [x] Default: `Porto/dataset/train_sorted.csv`, `--sorted` True

---

## Left (user-defined / TBD)

### 1. `update_db_from_vehicle_infos(db, vehicle_infos)`
**Location**: `scripts/csv_to_steps.py`

**Description**: Called when flushing a timestamp batch. Must apply vehicle info to the DB.

**To implement**:
- Update each vehicle in `db.vehicles` with dynamic fields from VehicleInfo (current_x, current_y, current_edge, current_position, speed, route_left, route_length_left, etc.)
- Remove vehicle from previous edge, add to current edge (`add_vehicle_to_edge`, `remove_vehicle_from_edge`)
- Update `road_edges` `vehicles_on_road`, `density`, `avg_speed`

---

### 2. `_update_edge_demand(db)`
**Location**: `scripts/csv_to_steps.py`

**Description**: Update `edge_demand` for all road edges before emitting JSON.

**To define**:
- Formula or algorithm for edge_demand
- How/when roads in active vehicles' routes contribute to demand

---

### 3. `_calculate_eta_and_labels(db, snapshot_timestamp)`
**Location**: `scripts/csv_to_steps.py`

**Description**: Calculate and add ETA label and other labels for vehicles at snapshot time.

**To define**:
- ETA label: formula or method
- Other labels: list and format
- Where to attach labels (vehicle node, etc.)

---

### 4. Optional / polish
- [ ] **Final step**: Emit JSON after last trajectory if there are remaining boundaries
- [ ] **current_zone**: Populate from network (e.g. edge zone or TAZ) if available
- [ ] **route_length_left** on vehicle node: ensure it matches VehicleInfo when building step JSON
- [ ] **suppress Pyproj warning**: `⚠️ Pyproj not available, using linear interpolation` (from NetworkParser)

---

## Summary

| Category | Done | Left |
|----------|------|------|
| Pipeline structure | 6/6 steps | 0 |
| Processing flow | 100% | 0 |
| **User-defined** | 0 | 3 (update_db, edge_demand, ETA/labels) |
| Polish | 0 | 3 (final step, zone, warnings) |
