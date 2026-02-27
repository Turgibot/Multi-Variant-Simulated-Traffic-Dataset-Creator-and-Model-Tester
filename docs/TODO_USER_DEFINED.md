# User-Defined Items (TBD)

Items to be defined by the user for the CSV → Step pipeline.

---

## 1. edge_demand

**Location**: `road_edges` in TrafficDB; `_update_edge_demand()` in `scripts/csv_to_steps.py`

**Description**: A score that depicts how and when a road is in the active vehicles' predefined routes.

**Current state**: Initialized to `0.0` in `build_edges_map()`. `_update_edge_demand(db)` is a placeholder (no-op).

**To define**:
- Formula or algorithm for computing edge_demand
- When to update it (e.g. at each snapshot, or incrementally as vehicles move)
- Units and range of the score

---

## 2. Update step (placeholder in snapshot loop)

**Location**: Main loop in `scripts/csv_to_steps.py`, inside `while ts >= snapshot_timestamp + sampling_period`

**Description**: User wrote: "update the (I will define it later)" — an update step to run before emitting each snapshot.

**Current state**: Not implemented; placeholder for user definition.

**To define**:
- What to update (e.g. edge_demand, other derived fields)
- Logic for the update

---

## 3. ETA label and other labels

**Location**: `_calculate_eta_and_labels()` in `scripts/csv_to_steps.py`

**Description**: Calculate and add ETA label and other labels for vehicles at snapshot time.

**Current state**: Placeholder (no-op).

**To define**:
- ETA label: formula or method (e.g. remaining distance / avg_speed, model-based)
- Other labels: list and format
- Where to attach labels (vehicle node, separate structure, etc.)

---

## Summary

| # | Item | Status |
|---|------|--------|
| 1 | edge_demand | TBD |
| 2 | Update step (snapshot loop) | TBD |
| 3 | ETA label and other labels | TBD |
