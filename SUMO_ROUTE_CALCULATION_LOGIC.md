# SUMO Route Calculation Logic

## Overview

This document describes the GPS-validated SUMO routing algorithm used to convert GPS trajectories into SUMO edge sequences. The algorithm combines SUMO's optimal routing with GPS trajectory validation to produce routes that are both SUMO-valid (connected, drivable) and GPS-validated (match actual trajectory).

## Core Assumptions

1. **Route Always Exists**: There is always a route between start and last GPS points, and SUMO must find it.
2. **GPS Inaccuracy**: GPS points are not accurate by nature. When looking for an edge that corresponds to a point, the selected edge must minimize the difference to the actual GPS points route.
3. **Base ID Matching Only**: **Always compare edges by base ID only (ignoring lane #)**. Lane-specific attributes (`#0`, `#1`, `#2`, etc.) are always ignored. This is because:
   - GPS accuracy doesn't allow distinguishing between lanes
   - Multiple lane variants represent the same road segment
   - Base ID matching handles GPS inaccuracy better than exact matching

## Algorithm Flow

### Step 0: Prepare Grid-Based Spatial Index

**Purpose**: Build spatial index for fast edge lookups (performed once when network loads).

**Process**:
1. Divide network into grid cells (500m × 500m default)
2. For each edge in network:
   - Calculate bounding box (min_x, min_y, max_x, max_y)
   - Add edge to all grid cells it intersects
   - Store edge bounding box for quick filtering
3. **Important**: Lane # is always ignored - edges are indexed by base edge ID only
   - `edge#0`, `edge#1`, `edge#2` → all treated as same base edge
   - Deduplicate during indexing to avoid redundant entries

**Output**: Spatial index ready for fast candidate edge queries

**Performance**:
- Build time: ~1-2 seconds for 277K edges
- Query time: O(k) where k = edges in nearby cells (typically 10-50 edges)
- Speedup: ~9,000x faster than iterating all edges

---

### Step 1: Initial SUMO Route Calculation

**Purpose**: Get a baseline route from start to destination using SUMO's shortest path algorithm.

**Note**: See "Core Assumptions" section above for fundamental assumptions that apply to all steps.

**Process**:

**Step 1.1: Find Candidate Edges for Start and Destination**
- **For start GPS point**:
  - Use spatial index to get candidate edges within 100m radius of GPS point
  - Calculate precise distance from GPS point to each candidate edge
  - Deduplicate by base edge ID (ignore lane #) - keep only closest variant per base edge
  - Return all candidate edges within 100m radius (sorted by distance, closest first)
  - If no candidates found within 100m: Return empty list
- **For destination GPS point**:
  - Use spatial index to get candidate edges within 100m radius of GPS point
  - Calculate precise distance from GPS point to each candidate edge
  - Deduplicate by base edge ID (ignore lane #) - keep only closest variant per base edge
  - Return all candidate edges within 100m radius (sorted by distance, closest first)
  - If no candidates found within 100m: Return empty list
- **Note**: Always use base edge ID only - lane # is always ignored (core assumption)
- Using 100m radius (instead of fixed 3 candidates) accounts for GPS inaccuracy and network density variations

**Step 1.2: Calculate K-Shortest Paths for All Combinations**
- For each start candidate edge and each destination candidate edge:
  - Calculate k-shortest paths (up to 5 routes) using `sumo_net.getKShortestPaths(start_candidate_edge, dest_candidate_edge, 5)`
  - If k-shortest paths not available: Fallback to single shortest path using `sumo_net.getShortestPath()`
  - If route found: Store route result with its cost
  - If route not found (None or empty): Skip this combination (log warning)
- Total routes: up to (N_start_candidates × N_dest_candidates × 5) routes
  - Where N_start_candidates = number of edges within 100m of start GPS point
  - Where N_dest_candidates = number of edges within 100m of destination GPS point
  - Typically fewer due to deduplication and failed combinations
- **Note**: Some combinations may fail - only valid routes are considered in Step 1.3

**Step 1.3: Select Top Routes and Calculate Similarity Scores**
- **Step 1.3.1: Get Top 5 Unique Routes**
  - Sort all routes by cost (ascending)
  - Deduplicate routes (same edge sequence) and take top 5 unique routes
  - This gives us up to 5 candidate routes to evaluate
  
- **Step 1.3.2: Calculate Similarity Score for Each Route**
  - For each of the top 5 routes, calculate a similarity score based on how well it matches the GPS trajectory
  - **Similarity Score Calculation** (Option 2: Weighted Point-to-Edge Distance with Edge Coverage):
    
    **Coverage Score (60% weight)**:
    - For each GPS point in the trajectory:
      - Find 3 closest candidate edges (using spatial index)
      - Check if any candidate's base ID (ignoring lane #) is in the route
      - Count how many GPS points have at least one candidate edge in the route
    - Coverage score = matched_points / total_points (0-1)
    
    **Distance Score (40% weight)**:
    - For each GPS point in the trajectory:
      - Calculate minimum distance from GPS point to any edge in the route
      - Use point-to-segment distance calculation for each edge shape
    - Average distance = sum(min_distances) / total_points
    - Distance score = exp(-average_distance / 100.0) (0-1, using exponential decay)
    
    **Final Similarity Score**:
    - Similarity = 0.6 × coverage_score + 0.4 × distance_score
    - Higher score (closer to 1.0) = better match with GPS trajectory
  
- **Step 1.3.3: Select Route with Highest Similarity Score**
  - Sort routes by similarity score (descending)
  - If multiple routes have the same similarity score: Use cost as tiebreaker (lower cost is better)
  - Select the route with the highest similarity score
  - This ensures we pick the route that best matches the GPS trajectory, not just the shortest route
  - Store selected route as `current_route_edges` (list of edge IDs)
  - Log all route scores for debugging
- If no valid routes found: Return empty route (handled in error handling)

**Step 1.4: Store Destination Candidate Edges**
- **Store up to 3 destination candidate edges** from Step 1.1 for later reuse
- These will be used in Step 3.3 when recalculating routes
- Store as: `destination_candidate_edges = [dest_edge_1, dest_edge_2, dest_edge_3]` (base IDs only)

**Step 1.5: Check Route Quality and Iterative Shortening**
- **Check if route found and similarity score ≥ 0.75**:
  - If yes: Route is acceptable, proceed to Step 1.6
  - If no: Proceed to iterative shortening (Step 1.5.1)

**Step 1.5.1: Iterative Shortening (if needed)**
- **Purpose**: Improve route quality by removing problematic GPS points from start/end
- **Trigger conditions**:
  - No route found in Step 1.3, OR
  - Rank 1 similarity score < 0.75
- **Process** (max 10 iterations, or until < 4 GPS points remain):
  - **Each iteration**:
    - Remove 6 GPS points from the start of the segment → Recalculate Step 1.1-1.3
    - If score ≥ 0.75 or no route found: Stop or continue
    - Remove 4 GPS points from the end of the segment → Recalculate Step 1.1-1.3
    - If score ≥ 0.75 or no route found: Stop or continue
    - Repeat with next iteration
  - **Stop conditions**:
    - Similarity score ≥ 0.75 (success)
    - Max 10 iterations reached
    - Number of GPS points < 4
- **Logging**: Log which GPS points were removed and why (no route found or low score)
- **Result**:
  - If similarity score ≥ 0.75: Use shortened segment and route
  - If still < 0.75 after iterations: Use best route found (highest score), Step 1 has failed, proceed to Step 2

**Step 1.6: Mark First Edge as Valid**
- Mark the first edge of the selected route as valid
- This edge will be used as the starting point for GPS validation
- **Note**: The final segment used may be shorter than the original (if iterative shortening occurred)

**Output**: 
- Initial route (e.g., 41 edges from start to destination) - the route with highest similarity score among top 5 candidate routes
- Final segment used (may be shortened from original if iterative shortening occurred)
- If similarity score ≥ 0.75: Step 1 succeeded, route is ready for use
- If similarity score < 0.75: Step 1 failed, will proceed to Step 2 (GPS validation loop)

**Note**: The similarity scoring ensures the selected route matches the GPS trajectory better than simply choosing the shortest path. This is important because GPS trajectories may not follow the absolute shortest path (e.g., due to traffic, preferences, or road conditions).

**Error Handling**:
- If no candidates found: Log error and return empty route
- If no route found for any combination: Proceed to iterative shortening (Step 1.5.1)
- If route calculation fails for all combinations: Proceed to iterative shortening (Step 1.5.1)
- If iterative shortening fails: Return best route found (if any), Step 1 has failed, proceed to Step 2

---

### Step 2: Initialize Validated Route

**Purpose**: Start building the validated route with the first edge from the selected route.

**Process**:
1. Get first edge from selected route (`current_route_edges[0]`)
2. Mark this edge as valid
3. Initialize `valid_route_edges = [current_route_edges[0]]`
4. Set `last_valid_edge_id = current_route_edges[0]`
5. Set `last_valid_gps_idx = 0`

**Note**: The first edge is from the route with highest similarity score selected in Step 1, not necessarily the edge mapped from the first GPS point. If iterative shortening occurred in Step 1, the first GPS point may have been removed.

---

### Step 3: GPS Point Validation Loop

For each GPS point `p` in the trajectory (starting from index 1):

#### Step 3.1: Find Candidate Edges

**Purpose**: Find 3 closest candidate edges to GPS point (using spatial index).

**Process**:
1. Convert GPS coordinates (lon, lat) to SUMO coordinates (x, y)
2. Use spatial index (from Step 0) to get candidate edge IDs (fast bounding box check)
3. For each candidate edge:
   - Get edge shape from first lane
   - Calculate minimum distance from GPS point to edge segments
4. Deduplicate by base edge ID (ignore lane #) - keep only closest variant per base edge
5. Sort candidates by distance
6. **Return top 3 closest candidates** (not all edges within 100m)

**Spatial Index**:
- Uses grid-based spatial index prepared in Step 0
- Grid cells: 500m × 500m (default)
- Indexes all edges by bounding boxes
- Reduces search from O(n) to O(k) where k << n

**Output**: List of 3 `(edge_id, distance)` tuples sorted by distance (closest first)

**Note**: 
- Lane # is always ignored - only base edge ID matters
- Returns exactly 3 candidates (or fewer if not enough edges found)
- No distance threshold (100m) - just returns closest 3

---

#### Step 3.2: Check if Candidate is in Route

**Purpose**: Validate if GPS point is on the calculated route.

**Process**:

**Step 3.2.1: Check for Any Match**
- Find position of `last_valid_edge_id` in `current_route_edges` → `last_valid_idx`
- For each of the 3 candidate edges:
  - **Always use base ID matching**: Strip `#` suffix to get candidate's base ID
  - **Compare only with edges at or after last valid position**: Compare candidate's base ID with base IDs of edges in `current_route_edges[last_valid_idx:]` (only edges from last valid position onwards)
  - This ensures we only look forward in the route, not backward
  - **Note**: If multiple route edges have the same base ID, match with the first occurrence (closest to last_valid_idx)
- If no candidate's base ID matches any route edge's base ID (from last valid position onwards): Proceed to Step 3.3 (recalculate route)
- If one or more candidates match: Proceed to Step 3.2.2

**Step 3.2.2: Handle Matches (Single or Multiple)**
- If one or more candidates match the route (from last valid position onwards):
  
  **Priority 1: Same Edge as Last Valid**
  - Check if any matching candidate has the same base ID as `last_valid_edge_id`
  - If yes: Use that candidate (vehicle is still on the same edge)
  - This handles cases where GPS point is still on the same road segment
  
  **Priority 2: Closest to Last Valid Edge (by Route Index)**
  - If no candidate matches last valid edge (or only one candidate matches):
    - For each matching candidate, find its position in route (`candidate_idx`)
    - All candidates are already at or after `last_valid_idx` (from Step 3.2.1)
    - Select the candidate with the smallest `candidate_idx - last_valid_idx` (closest forward progress)
    - This ensures we progress forward along the route
    - If only one candidate matches: Use that candidate


**Step 3.2.3: Progress Route Forward**
- Find position of selected matched edge in route (`matched_idx`)
  - **Note**: If multiple route edges have the same base ID as matched edge, use the first occurrence at or after `last_valid_idx`
- Find position of last valid edge (`last_valid_idx`)
- If `matched_idx > last_valid_idx` (progressing forward):
  - Add all intermediate edges from `last_valid_idx + 1` to `matched_idx` to `valid_route_edges`
  - Update `last_valid_edge_id = matched_edge_id`
  - Update `last_valid_gps_idx = gps_idx`
  - Continue to next GPS point
- If `matched_idx == last_valid_idx` (same position):
  - Vehicle is still on the same edge (from Priority 1)
  - Update `last_valid_gps_idx = gps_idx` (update GPS index but not edge)
  - Continue to next GPS point
- If `matched_idx < last_valid_idx` (shouldn't happen):
  - Log warning (edge is before last valid position - this shouldn't occur due to Step 3.2.1 filtering)
  - Continue to next GPS point (no progress)

**Matching Strategy**:

- **Base ID Matching Only**: Always strip `#` suffix and compare base IDs
  - Example: `"96151239#10"` → base ID `"96151239"`
  - Compare candidate's base ID with base IDs of edges in route
  - No exact matching - always use base ID comparison

**Note**: 
- Since we have 3 candidates, multiple candidates might match the route (by base ID)
- The selection logic (Priority 1 and Priority 2) ensures we pick the most appropriate one based on vehicle progress
- Lane # is always ignored - this is a core assumption of the algorithm

---

#### Step 3.3: Recalculate Route (GPS Point Not on Route)

**Purpose**: GPS point diverges from current route → recalculate using candidate edges from Step 3.1.

**Process**:

**3.3.1: Find Best Candidate Edge (Minimum Total Cost)**
- **Use the 3 candidate edges already found in Step 3.1** (no need to find new ones)
- For each of the 3 candidate edges (`candidate_edge`):
  
  **Step 1: Route from Last Valid Edge to Candidate Edge**
  - Calculate route: `sumo_net.getShortestPath(last_valid_edge, candidate_edge)`
  - Store route edges and cost: `(route_to_candidate, cost_to_candidate)`
  - If no route found: Skip this candidate (log warning)
  
  **Step 2: Route from Candidate Edge to Best Destination**
  - For each of the 3 destination candidate edges (stored in Step 1.4):
    - Calculate route: `sumo_net.getShortestPath(candidate_edge, dest_candidate_edge)`
    - If route found: Store route edges and cost: `(route_to_dest, cost_to_dest)`
    - If route not found: Skip this destination candidate (log warning)
  - Find the destination candidate with minimum cost: `best_dest_edge` with `min_cost_to_dest`
  - If multiple destinations have the same minimum cost: Select the first one found
  - If no route found to any destination: Skip this candidate (log warning)
  
  **Step 3: Calculate Total Cost**
  - Total cost for this candidate = `cost_to_candidate + min_cost_to_dest`
  - Store: `(candidate_edge, route_to_candidate, best_dest_edge, route_to_dest, total_cost)`

- **Select candidate with minimum total cost**
- If no valid candidate found (all failed in Step 1 or Step 2): Log error and skip to next GPS point

**3.3.2: Update Route with Selected Candidate**
- **Keep**: Edges from start to last valid edge: `current_route_edges[:last_valid_idx + 1]`
- **Add**: Route from last valid edge to found candidate edge: `route_to_candidate`
  - **Note**: `route_to_candidate` should start with `last_valid_edge` (or connect to it)
  - If `route_to_candidate[0] == last_valid_edge_id`: Skip first edge (already in route)
  - If `route_to_candidate` is empty (direct connection): This shouldn't happen, but handle gracefully
- **Add**: Route from found candidate edge to best destination edge: `route_to_dest`
  - **Note**: `route_to_dest` should start with `found_edge` (or connect to it)
  - If `route_to_dest[0] == found_edge`: Include full route
  - If `route_to_dest` is empty (direct connection): This shouldn't happen, but handle gracefully
- **Updated route**: `current_route_edges = [start...last_valid] + [last_valid...found_edge] + [found_edge...best_dest]`

**3.3.3: Mark Edges as Valid**
- Mark all edges in `route_to_candidate` (from last_valid_edge to found_edge) as valid
- Add all edges from `route_to_candidate` to `valid_route_edges`
  - Skip `last_valid_edge` if it's already in `valid_route_edges`
  - Include `found_edge` (last edge of `route_to_candidate`)
- Update `last_valid_edge_id = found_edge`
- Update `last_valid_gps_idx = gps_idx`

---

### Step 4: Return Validated Route

**Output**: `valid_route_edges` - List of validated edge IDs

**Properties**:
- All edges are connected (SUMO-valid)
- Route follows GPS trajectory (GPS-validated)
- Handles GPS inaccuracies (100m tolerance)
- Handles route deviations (recalculates when needed)

---

## Key Functions

### `_calculate_route_similarity_score(route_edges, gps_points, seg_idx)`

Calculates similarity score for a route based on how well it matches GPS trajectory.

**Parameters**:
- `route_edges`: List of edge IDs in the route
- `gps_points`: List of GPS points `[[lon, lat], ...]`
- `seg_idx`: Segment index for logging

**Returns**: Similarity score (0-1, higher is better match)

**Algorithm**:
- Uses Option 2: Weighted Point-to-Edge Distance with Edge Coverage
- Coverage score (60%): Percentage of GPS points with candidate edges in route
- Distance score (40%): Exponential decay of average distance from GPS points to route
- Final score: Weighted combination of both metrics

---

### `_build_gps_validated_sumo_route(segment, start_edge_id, dest_edge_id, seg_idx)`

Main function that implements the algorithm.

**Parameters**:
- `segment`: List of GPS points `[[lon, lat], ...]`
- `start_edge_id`: Edge ID from first GPS point
- `dest_edge_id`: Edge ID from last GPS point
- `seg_idx`: Segment index for logging

**Returns**: List of validated edge IDs

**Internal State**:
- `destination_candidate_edges`: List of 3 destination candidate edges (stored in Step 1.4, reused in Step 3.3)

---

### `_find_candidate_edges_in_radius(lon, lat, max_candidates=3)`

Finds 3 closest candidate edges to GPS coordinates (using spatial index).

**Process**:
1. Convert GPS to SUMO coordinates
2. Use spatial index (from Step 0) for fast lookup
3. Calculate precise distances for candidates
4. **Always deduplicate by base edge ID** (ignore lane #) - core assumption
5. Sort by distance
6. Return top 3 closest candidates

**Returns**: List of up to 3 `(edge_id, distance)` tuples sorted by distance (closest first)

**Note**: 
- No distance threshold - returns closest edges regardless of distance
- **Always uses base edge ID only** - lane # is never considered (core assumption)

---

### `_get_base_edge_id(edge_id)`

Strips `#` suffix from edge ID.

**Examples**:
- `"1016528107#1"` → `"1016528107"`
- `"1016528107#0"` → `"1016528107"`
- `"1016528107"` → `"1016528107"`
- `"-1016528107#1"` → `"-1016528107"`

**Purpose**: Handle lane/direction variants as same base edge

---

## Data Structures

### `current_route_edges`
- Type: `List[str]`
- Content: Current calculated route (may be updated during recalculation)
- Updated: When route is recalculated in Step 3.3

### `valid_route_edges`
- Type: `List[str]`
- Content: Validated edges that match GPS trajectory
- Updated: When GPS points match route edges

### `last_valid_edge_id`
- Type: `str`
- Content: Last edge that was validated by GPS point
- Updated: When a match is found or route is recalculated

### `last_valid_gps_idx`
- Type: `int`
- Content: Index of last GPS point that validated an edge
- Updated: When a match is found or route is recalculated

### `destination_candidate_edges`
- Type: `List[str]`
- Content: Up to 3 destination candidate edges (base IDs) from Step 1.1
- Stored: In Step 1.4 after initial route calculation
- Reused: In Step 3.3.1 when recalculating routes

### `final_segment`
- Type: `List[List[float]]`
- Content: Final GPS segment used for route calculation (may be shortened from original)
- Updated: During Step 1.5.1 (iterative shortening) if needed
- Used: For star visualization - only GPS points in final_segment get stars

---

## Spatial Index

### `EdgeSpatialIndex`

Grid-based spatial index for fast edge lookups (prepared in Step 0).

**Initialization** (Step 0):
- Divides network into 500m × 500m grid cells
- Indexes each edge by grid cells it intersects
- Stores bounding boxes for quick filtering
- **Important**: Lane # is always ignored during indexing - edges deduplicated by base edge ID

**Query**:
- `get_candidates_in_radius(x, y, radius)` → Returns set of edge IDs that might be within radius
- Uses bounding box check (fast)
- Precise distance calculation done afterwards
- Results are deduplicated by base edge ID (ignoring lane #)

**Usage**:
- Step 1.1: Find all candidate edges within 100m radius for start/destination GPS points
- Step 3.1: Find 3 closest candidates for each GPS point in trajectory
- Step 3.3.1: Reuses candidates from Step 3.1 (no new spatial index query needed)

**Performance**:
- Build time: ~1-2 seconds for 277K edges (done once when network loads)
- Query time: O(k) where k = edges in nearby cells (typically 10-50 edges)
- Speedup: ~9,000x faster than iterating all edges

---

## Edge Cases Handled

1. **GPS Point Not Mappable**: Skip GPS point, continue to next
2. **No Candidates Found**: If no candidates found within 100m for start/destination, proceed to iterative shortening
3. **No Route Found for Any Combination**: Proceed to iterative shortening (Step 1.5.1)
4. **Low Similarity Score**: If similarity score < 0.75, proceed to iterative shortening (Step 1.5.1)
5. **Iterative Shortening Limits**:
   - If < 4 GPS points remain: Stop shortening, use best route found
   - If max 10 iterations reached: Stop shortening, use best route found
   - If similarity score ≥ 0.75: Success, use shortened segment
6. **Edge Not in Network**: Log diagnostic, try to find similar edges
7. **Route Recalculation Fails (Step 3.3)**:
   - If no route found from last_valid to candidate: Skip candidate, try next
   - If no route found from candidate to any destination: Skip candidate, try next
   - If all candidates fail: Log error, skip GPS point, continue to next
8. **Multiple Edge Variants**: Deduplicate by base ID, keep closest (ignores lane #)
9. **GPS Point Before Last Valid**: Log info, don't progress backward
10. **Multiple Routes with Same Cost**: Select first route found (or could use additional criteria)
11. **No Valid Candidate in Step 3.3**: Log error, skip GPS point, continue to next GPS point
12. **Step 1 Failure**: If Step 1 fails (similarity < 0.75 after iterative shortening), proceed to Step 2

---

## Configuration Parameters

- **Spatial Index Grid Cell Size**: 500 meters (Step 0 - prepared once when network loads)
- **Start/End Candidate Radius**: 100 meters (Step 1.1 - all edges within radius)
- **Intermediate Candidate Count**: 3 candidates per GPS point (Step 3.1)
- **Lane # Handling**: Always ignored - edges deduplicated by base edge ID
- **Base ID Matching**: Enabled (handles `#` suffixes, ignores lane attributes)
- **K-Shortest Paths**: Up to 5 routes per start/destination combination (Step 1.2)
- **Top Routes for Scoring**: Up to 5 unique routes evaluated for similarity (Step 1.3)
- **Similarity Scoring Weights**: 60% coverage, 40% distance (Step 1.3.2)
- **Distance Scale Factor**: 100 meters (for exponential decay in distance score)
- **Similarity Threshold**: 0.75 (Step 1.5 - minimum score for Step 1 success)
- **Iterative Shortening**: Max 10 iterations, minimum 4 GPS points (Step 1.5.1)
  - **Removal Pattern**: 6 points from start, then 4 points from end, repeating

---

## Example Flow

```
Step 0: Prepare Spatial Index (done once when network loads)
  - Build grid-based spatial index
  - Index all edges (ignoring lane #)
  - Ready for fast candidate queries

Step 1: Initial Route Calculation
  P0 (start): Use spatial index → Find all candidate edges within 100m → A, A', A'', A''', ...
  P7 (dest): Use spatial index → Find all candidate edges within 100m → H, H', H'', H''', ...
  Calculate k-shortest paths for all combinations (up to 5 routes per combination):
    Route 1: A → H (cost: 1000, similarity: 0.75)
    Route 2: A → H' (cost: 1200, similarity: 0.82) ← Highest similarity
    Route 3: A → H'' (cost: 1500, similarity: 0.68)
    Route 4: A' → H (cost: 1100, similarity: 0.71)
    Route 5: A' → H' (cost: 1050, similarity: 0.79)
    ... (more routes from all combinations)
  Top 5 unique routes selected, similarity scores calculated
  Selected: Route 2 (A → H') ✅ (highest similarity score: 0.82 ≥ 0.75)
  
  If similarity < 0.75 or no route found:
    Iteration 1: Remove P0-P5 (6 from start) → Recalculate
    Iteration 1: Remove P6-P9 (4 from end) → Recalculate
    Iteration 2: Remove P6-P11 (6 from start) → Recalculate
    Iteration 2: Remove P12-P15 (4 from end) → Recalculate
    ... (continue until score ≥ 0.75 or max iterations/points reached)

Initial Route: A' → B → C → D → E → F → G → H'
GPS Points:    P0  P1  P2  P3  P4  P5  P6  P7

P0: Start edge A' ✅ (marked valid - from selected route)
P1: Use spatial index → Find 3 closest candidates → B, B', B''
    B found in route ✅ → Adds B to valid route
P2: Use spatial index → Find 3 closest candidates → C, C', C''
    C found in route ✅ → Adds C to valid route
P3: Use spatial index → Find 3 closest candidates → X, X', X''
    None found in route (from last valid C onwards) ⚠️
    → For each candidate (X, X', X''):
      → Calculate route: C → candidate (cost1)
      → Calculate route: candidate → best_dest (cost2)
      → Total cost = cost1 + cost2
    → Select candidate with minimum total cost: X
    → Update route: [A'...C] + [C...X] + [X...H']
    → Mark edges C to X (inclusive) as valid
    → New route: A' → B → C → X → Y → Z → H'
P4: Finds edge Y in new route ✅ → Adds Y to valid route
P5: Finds edge Z in route ✅ → Adds Z to valid route
P6: Finds edge H' in route ✅ → Adds H' to valid route
P7: Destination edge H' ✅ → Already in route

Final Route: A' → B → C → X → Y → Z → H'
```

---

## Notes for Modification

When modifying this logic, consider:

1. **Spatial Index Grid Size**: Currently 500m cells - may need tuning for different network densities
2. **Candidate Count**: Currently 3 candidates per GPS point - may need adjustment based on GPS accuracy and network density
3. **Lane # Handling**: Currently always ignored - this is intentional to handle GPS inaccuracy
4. **Base ID Matching**: Currently enabled - may want exact matching only
5. **Route Selection**: Currently uses minimum cost - may want to consider other factors (e.g., route length, number of turns)
6. **Recalculation Strategy**: Currently replaces route from GPS point - may want different approach
7. **Edge Filtering**: Currently includes all edges - may want to filter by vehicle class
8. **Distance Threshold**: Currently none - may want to add maximum distance threshold for candidates

---

## Current Implementation Location

File: `src/gui/dataset_conversion_page.py`

Key Methods:
- `_build_gps_validated_sumo_route()` - Main algorithm (line ~3758)
- `_calculate_step1_route()` - Step 1 route calculation with iterative shortening (line ~3231)
- `_calculate_step1_route_single()` - Single Step 1 calculation attempt (line ~3083)
- `_calculate_route_similarity_score()` - Similarity score calculation (line ~3025)
- `_find_candidate_edges()` - Candidate edge lookup with radius support (line ~2869)
- `_get_base_edge_id()` - Base ID extraction (line ~2805)
- `EdgeSpatialIndex` - Spatial index class (line ~44)

