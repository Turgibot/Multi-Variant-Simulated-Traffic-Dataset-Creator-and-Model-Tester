# Road and Junction Assignment Logic

## Core Principles

1. **Junctions**: One-to-one assignment (assigned to FIRST zone where they're found)
   - Once assigned, cannot be reassigned

2. **Roads**: One-to-one assignment (assigned to FIRST zone where they have shape points)
   - Once assigned, cannot be reassigned
   - A road can have its start junction in Zone A and end junction in Zone B, but the road itself is assigned to only ONE zone based on its shape points

## Assignment Flow (when a zone area is selected)

### Step 1: Get junctions and roads already assigned to other zones
```python
assigned_junctions, assigned_roads = self._get_assigned_junctions_and_roads_for_other_zones(zone_id)
```
- Scans all other zones
- Returns:
  - Set of junction IDs already assigned
  - Set of road IDs already assigned
- Both follow one-to-one assignment (first zone wins)

### Step 2: Collect nodes (including junctions) in current zone
- Single set: `nodes_in_zone_all_areas` (contains both regular nodes and junctions)
- For each rectangle in `zone_areas`:
  - Regular nodes: if point is in rectangle → add to set
  - Junctions: if point is in rectangle AND not in `assigned_junctions` → add to set
- Junctions are skipped if already assigned (one-to-one rule)

### Step 3: Assign roads based on shape points
- For each road:
  1. **Skip if already assigned** to another zone (check `assigned_roads`)
  2. Check ALL shape points from ALL lanes
  3. If ANY shape point is in ANY zone rectangle → assign road to this zone
  4. Once assigned, road cannot be reassigned (one-to-one rule)
  5. Fallback: if no shape points, check endpoints (from_node/to_node in `nodes_in_zone_all_areas`)

## Important Points

- **Roads are one-to-one**: A road can only be assigned to ONE zone
- **Junctions are one-to-one**: Once assigned to Zone A, they won't be assigned to Zone B
- **Assignment order**: When checking which zone a road belongs to, zones are processed in the order they exist in the system (typically creation order)
- **"First zone" means**: The first zone (in processing order) that finds shape points of the road will assign it to itself
- **Roads don't depend on junctions**: A road connecting to a junction in Zone A can still be assigned to Zone B if its shape points are in Zone B (but only if Zone B is processed before Zone A)
- **Shape points take priority**: Roads are assigned based on geometry (shape points), not endpoints
- **Once assigned, never reassigned**: Once a road is assigned to a zone, it cannot be reassigned to another zone

## Example Scenario

- Row 1 selected → junctions J1, J2 assigned to Row 1
- Row 2 selected → junctions J3, J4 assigned to Row 2
- Vertical road V connects J1 (Row 1) to J3 (Row 2):
  - V has shape points in both Row 1 and Row 2 zones
  - If Row 1 is processed first: V gets assigned to Row 1 (first shape point found)
  - When Row 2 is processed: V is skipped (already assigned to Row 1)
  - Result: V is assigned to Row 1 only, even though it connects to J3 in Row 2
  - Junction assignments don't affect road assignments - roads are assigned based on shape points

## Key Difference from Previous Logic

**Previous (incorrect)**: Roads could be assigned to multiple zones
**Current (correct)**: Roads follow one-to-one assignment, just like junctions
