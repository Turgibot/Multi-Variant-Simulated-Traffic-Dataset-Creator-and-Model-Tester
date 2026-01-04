# Edge Matching Logic - EXACT STEP-BY-STEP PROCESS

## Function: `_find_edges_matching_gps_direction(gps_point1, gps_point2)`

### INPUT:
- `gps_point1`: Tuple `(x1, y1_flipped)` - GPS point 1 in **Y-FLIPPED SUMO coordinates**
- `gps_point2`: Tuple `(x2, y2_flipped)` - GPS point 2 in **Y-FLIPPED SUMO coordinates**

### STEP 1: Setup Y-Flip Function
```python
y_min = getattr(self.map_view, '_network_y_min', 0)
y_max = getattr(self.map_view, '_network_y_max', 0)

def flip_y(y):
    return y_max + y_min - y
```

### STEP 2: Calculate GPS Direction
```python
x1, y1_flipped = gps_point1  # Already Y-flipped
x2, y2_flipped = gps_point2  # Already Y-flipped

dx_gps = x2 - x1
dy_gps = y2_flipped - y1_flipped
gps_angle = atan2(dy_gps, dx_gps)  # GPS direction angle
```

### STEP 3: Get All Edges from Network Parser
```python
edges = self.network_parser.get_edges()
# Returns edges with shape points in ORIGINAL SUMO coordinates (NOT Y-flipped)
```

### STEP 4: Filter to Vehicle-Allowed Edges in Bounding Box
For each edge:
1. Check `edge_data.get('allows_passenger', True)` → keep only vehicle-allowed edges
2. Get `shape_points = edge_data['lanes'][0]['shape']` → **ORIGINAL SUMO coordinates**
3. For each shape point:
   - `x, y_sumo = point[0], point[1]` → **ORIGINAL**
   - `y = flip_y(y_sumo)` → **Y-FLIP IT**
   - Check if `(x, y)` is inside `_bounding_box_polygon`
4. For each edge segment:
   - `x1, y1_sumo = shape_points[i]` → **ORIGINAL**
   - `x2, y2_sumo = shape_points[i+1]` → **ORIGINAL**
   - `y1 = flip_y(y1_sumo)`, `y2 = flip_y(y2_sumo)` → **Y-FLIP BOTH**
   - Check if segment `(x1, y1)` to `(x2, y2)` intersects bounding box
5. Store edge_id in `vehicle_edges_in_box` if edge is in box

### STEP 5: Calculate Distance from GPS Points to Each Edge
For each edge in `vehicle_edges_in_box`:
1. Get `shape_points = edge_data['lanes'][0]['shape']` → **ORIGINAL SUMO coordinates**
2. For each segment `i` in edge:
   ```python
   edge_x1, edge_y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL
   edge_x2, edge_y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL
   
   # Y-FLIP for distance calculation
   edge_y1 = flip_y(edge_y1_sumo)
   edge_y2 = flip_y(edge_y2_sumo)
   
   # Calculate distance (both GPS and edge are now Y-flipped)
   dist_gps1 = point_to_segment_distance(
       QPointF(x1, y1_flipped),  # GPS point 1 (Y-flipped)
       QPointF(edge_x1, edge_y1),  # Edge segment start (Y-flipped)
       QPointF(edge_x2, edge_y2)   # Edge segment end (Y-flipped)
   )
   
   dist_gps2 = point_to_segment_distance(
       QPointF(x2, y2_flipped),  # GPS point 2 (Y-flipped)
       QPointF(edge_x1, edge_y1),  # Edge segment start (Y-flipped)
       QPointF(edge_x2, edge_y2)   # Edge segment end (Y-flipped)
   )
   
   min_distance = min(min_distance, dist_gps1, dist_gps2)
   ```

### STEP 6: Check Direction Match
For each edge:
1. Get `from_node_id` and `to_node_id` from `edge_data`
2. Get node coordinates:
   ```python
   from_node = nodes.get(from_node_id)  # ORIGINAL SUMO coordinates
   to_node = nodes.get(to_node_id)       # ORIGINAL SUMO coordinates
   
   edge_start_x = from_node['x']
   edge_start_y_sumo = from_node['y']  # ORIGINAL
   edge_end_x = to_node['x']
   edge_end_y_sumo = to_node['y']      # ORIGINAL
   
   # Y-FLIP for direction calculation
   edge_start_y = flip_y(edge_start_y_sumo)
   edge_end_y = flip_y(edge_end_y_sumo)
   ```
3. Calculate edge direction:
   ```python
   edge_dx = edge_end_x - edge_start_x
   edge_dy = edge_end_y - edge_start_y  # Using Y-flipped coordinates
   edge_angle = atan2(edge_dy, edge_dx)
   ```
4. Calculate angle difference:
   ```python
   angle_diff = abs(gps_angle - edge_angle)
   if angle_diff > π:
       angle_diff = 2π - angle_diff
   ```
5. Check if `angle_diff <= π/4` (45 degrees) → if yes, add to `matching_edges`

### STEP 7: Select Closest Edge
```python
closest_edge = min(matching_edges, key=lambda x: x[1])  # x[1] is min_distance
closest_edge_id = closest_edge[0]
```

### STEP 8: Draw the Closest Edge
1. Get `shape_points = edge_data['lanes'][0]['shape']` → **ORIGINAL SUMO coordinates**
2. For each segment:
   ```python
   x1_edge, y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL
   x2_edge, y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL
   
   # Y-FLIP for drawing
   y1 = flip_y(y1_sumo)
   y2 = flip_y(y2_sumo)
   
   # Draw line
   scene.addLine(x1_edge, y1, x2_edge, y2, purple_pen)
   ```

## SUMMARY OF COORDINATE SYSTEMS:

1. **GPS Points**: Already Y-flipped when passed to function ✓
2. **Edge Shape Points from `network_parser.get_edges()`**: ORIGINAL SUMO coordinates (NOT Y-flipped)
3. **Edge Nodes from `network_parser.get_nodes()`**: ORIGINAL SUMO coordinates (NOT Y-flipped)
4. **Bounding Box Polygon**: Created from GPS points (Y-flipped coordinates)

## WHAT WE DO:
- We Y-flip edge shape points when checking bounding box ✓
- We Y-flip edge shape points when calculating distance ✓
- We Y-flip edge nodes when calculating direction ✓
- We Y-flip edge shape points when drawing ✓

## POTENTIAL ISSUE:
The distance calculation uses `point_to_segment_distance` which finds the closest point on the edge segment. This means:
- If an edge is very long, the closest point might be in the middle of the edge
- The edge endpoints might be far from GPS points
- But `min_distance` will be small if any point on the edge is close

This is why we see edges far from GPS points but with small `min_distance` values.

