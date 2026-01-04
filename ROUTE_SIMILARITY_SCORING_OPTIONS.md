# Route Similarity Scoring Options

## Overview

After Step 1 calculates 5 different routes from start to destination, we need to score each route based on how well it matches the GPS trajectory. The route with the highest similarity score will be selected.

## Proposed Approaches

### Option 1: Average Point-to-Edge Distance (Recommended)

**Concept**: For each GPS point, calculate the minimum distance to any edge in the route, then compute the average.

**Formula**:
```
score = 1 / (1 + average_distance)
similarity = 1 - min(1, average_distance / max_threshold)
```

**Process**:
1. For each GPS point in the trajectory:
   - Find the minimum distance from the GPS point to any edge in the route
   - Use spatial index to find candidate edges, then calculate precise distance
2. Calculate average of all minimum distances
3. Convert to similarity score (higher = better match)

**Pros**:
- Simple and intuitive
- Fast computation (can reuse spatial index)
- Handles GPS inaccuracy well
- Easy to tune with threshold

**Cons**:
- Doesn't consider route order/sequence
- May favor routes that pass near GPS points but in wrong order

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    total_distance = 0
    for lon, lat in gps_points:
        # Convert GPS to SUMO coordinates
        x, y = convert_gps_to_sumo(lon, lat)
        
        # Find minimum distance to any edge in route
        min_dist = float('inf')
        for edge_id in route_edges:
            edge = sumo_net.getEdge(edge_id)
            dist = calculate_point_to_edge_distance(x, y, edge)
            min_dist = min(min_dist, dist)
        
        total_distance += min_dist
    
    avg_distance = total_distance / len(gps_points)
    
    # Convert to similarity score (0-1, higher is better)
    # Using exponential decay: score = exp(-avg_distance / scale_factor)
    scale_factor = 100.0  # meters - tune based on GPS accuracy
    similarity = math.exp(-avg_distance / scale_factor)
    
    return similarity
```

---

### Option 2: Weighted Point-to-Edge Distance with Edge Coverage

**Concept**: Combine average distance with a coverage metric that rewards routes containing edges close to GPS points.

**Formula**:
```
coverage_score = (edges_in_route_matching_gps_candidates) / total_gps_points
distance_score = 1 / (1 + average_distance)
final_score = 0.6 * coverage_score + 0.4 * distance_score
```

**Process**:
1. For each GPS point:
   - Find 3 closest candidate edges (using spatial index)
   - Check if any of these candidates (by base ID) are in the route
   - Track coverage count
2. Calculate coverage: `coverage = matched_points / total_points`
3. Calculate average distance (same as Option 1)
4. Combine both metrics with weights

**Pros**:
- Considers both proximity and edge matching
- More robust to GPS noise
- Rewards routes that actually contain edges near GPS points

**Cons**:
- Slightly more complex
- Requires tuning weights (coverage vs distance)

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    route_base_ids = {edge_id.split('#')[0] for edge_id in route_edges}
    
    matched_points = 0
    total_distance = 0
    
    for lon, lat in gps_points:
        # Find 3 closest candidates
        candidates = find_candidate_edges(lon, lat, max_candidates=3)
        
        # Check if any candidate's base ID is in route
        candidate_base_ids = {edge_id.split('#')[0] for edge_id, _ in candidates}
        if route_base_ids.intersection(candidate_base_ids):
            matched_points += 1
        
        # Calculate minimum distance to route
        min_dist = float('inf')
        for edge_id in route_edges:
            edge = sumo_net.getEdge(edge_id)
            dist = calculate_point_to_edge_distance(x, y, edge)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    coverage = matched_points / len(gps_points)
    avg_distance = total_distance / len(gps_points)
    
    # Normalize scores
    coverage_score = coverage  # Already 0-1
    distance_score = math.exp(-avg_distance / 100.0)  # 0-1
    
    # Weighted combination
    final_score = 0.6 * coverage_score + 0.4 * distance_score
    
    return final_score
```

---

### Option 3: Sequential Edge Matching Score

**Concept**: Reward routes where GPS points match route edges in sequential order (not just any order).

**Formula**:
```
For each GPS point:
  - Find closest candidate edges
  - Check if any candidate matches the "expected" edge in route sequence
  - Track sequential matches

score = sequential_matches / total_points
```

**Process**:
1. Initialize route position tracker
2. For each GPS point:
   - Find 3 closest candidate edges
   - Check if any candidate matches the current or next edges in route (by base ID)
   - If match found: advance route position tracker
   - Count sequential matches
3. Score = sequential_matches / total_points

**Pros**:
- Considers route order/sequence
- More accurate for trajectories with clear progression
- Handles cases where route passes near GPS points but in wrong order

**Cons**:
- More complex to implement
- May be too strict for noisy GPS data
- Requires careful handling of route position tracking

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    route_base_ids = [edge_id.split('#')[0] for edge_id in route_edges]
    route_position = 0  # Current position in route
    sequential_matches = 0
    lookahead_window = 3  # Check next 3 edges ahead
    
    for lon, lat in gps_points:
        # Find 3 closest candidates
        candidates = find_candidate_edges(lon, lat, max_candidates=3)
        candidate_base_ids = {edge_id.split('#')[0] for edge_id, _ in candidates}
        
        # Check if any candidate matches current or next edges in route
        matched = False
        for offset in range(lookahead_window):
            if route_position + offset < len(route_base_ids):
                if route_base_ids[route_position + offset] in candidate_base_ids:
                    sequential_matches += 1
                    route_position = route_position + offset + 1  # Advance position
                    matched = True
                    break
        
        if not matched:
            # GPS point doesn't match expected route position
            # Don't advance route_position (or advance by 1 to avoid getting stuck)
            route_position = min(route_position + 1, len(route_base_ids) - 1)
    
    score = sequential_matches / len(gps_points)
    return score
```

---

### Option 4: Hausdorff Distance (Maximum Deviation)

**Concept**: Measure the maximum distance between GPS trajectory and route (one-sided Hausdorff distance).

**Formula**:
```
hausdorff_distance = max(min(distance(gps_point, route_edge) for edge in route) for gps_point in trajectory)
similarity = 1 / (1 + hausdorff_distance / scale_factor)
```

**Process**:
1. For each GPS point, find minimum distance to route
2. Take the maximum of all minimum distances
3. Convert to similarity score

**Pros**:
- Captures worst-case deviation
- Good for ensuring no GPS point is too far from route
- Simple to understand

**Cons**:
- Sensitive to outliers (single bad GPS point can ruin score)
- Doesn't consider overall route quality
- May not distinguish well between routes

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    max_min_distance = 0
    
    for lon, lat in gps_points:
        x, y = convert_gps_to_sumo(lon, lat)
        
        min_dist = float('inf')
        for edge_id in route_edges:
            edge = sumo_net.getEdge(edge_id)
            dist = calculate_point_to_edge_distance(x, y, edge)
            min_dist = min(min_dist, dist)
        
        max_min_distance = max(max_min_distance, min_dist)
    
    # Convert to similarity (penalize large maximum deviations)
    scale_factor = 200.0  # meters
    similarity = 1 / (1 + max_min_distance / scale_factor)
    
    return similarity
```

---

### Option 5: Frechet Distance (Curve Similarity)

**Concept**: Measure similarity between two curves (GPS trajectory vs route shape) using Frechet distance.

**Formula**:
```
frechet_distance = minimum leash length needed to walk both curves simultaneously
similarity = 1 / (1 + frechet_distance / scale_factor)
```

**Process**:
1. Convert route edges to polyline (route shape)
2. Convert GPS points to polyline
3. Calculate Frechet distance between polylines
4. Convert to similarity score

**Pros**:
- Most accurate for curve similarity
- Considers both shape and order
- Standard metric in computational geometry

**Cons**:
- Computationally expensive (O(n*m) for n GPS points, m route segments)
- Complex to implement correctly
- May be overkill for this use case

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    # Convert route to polyline
    route_polyline = []
    for edge_id in route_edges:
        edge = sumo_net.getEdge(edge_id)
        shape = edge.getShape()
        route_polyline.extend(shape)
    
    # Convert GPS to SUMO coordinates
    gps_polyline = [convert_gps_to_sumo(lon, lat) for lon, lat in gps_points]
    
    # Calculate Frechet distance (simplified version)
    frechet_dist = calculate_frechet_distance(gps_polyline, route_polyline)
    
    # Convert to similarity
    scale_factor = 150.0  # meters
    similarity = 1 / (1 + frechet_dist / scale_factor)
    
    return similarity
```

---

### Option 6: Hybrid Multi-Metric Score (Most Robust)

**Concept**: Combine multiple metrics with weights to get a comprehensive similarity score.

**Formula**:
```
coverage_score = edges_matching_gps_candidates / total_points
distance_score = exp(-avg_distance / 100)
sequential_score = sequential_matches / total_points
hausdorff_score = 1 / (1 + max_deviation / 200)

final_score = w1 * coverage_score + w2 * distance_score + w3 * sequential_score + w4 * hausdorff_score
```

**Process**:
1. Calculate all 4 metrics (coverage, distance, sequential, hausdorff)
2. Normalize each to 0-1 range
3. Combine with weights (e.g., 0.3, 0.3, 0.2, 0.2)

**Pros**:
- Most robust and comprehensive
- Handles various edge cases
- Can be tuned for specific use cases

**Cons**:
- Most complex to implement
- Requires tuning multiple weights
- Slower computation

**Implementation**:
```python
def calculate_similarity_score(route_edges, gps_points, spatial_index, sumo_net):
    # Calculate all metrics
    coverage_score = calculate_coverage_score(route_edges, gps_points, spatial_index)
    distance_score = calculate_distance_score(route_edges, gps_points, sumo_net)
    sequential_score = calculate_sequential_score(route_edges, gps_points, spatial_index)
    hausdorff_score = calculate_hausdorff_score(route_edges, gps_points, sumo_net)
    
    # Weighted combination
    weights = {
        'coverage': 0.3,
        'distance': 0.3,
        'sequential': 0.2,
        'hausdorff': 0.2
    }
    
    final_score = (
        weights['coverage'] * coverage_score +
        weights['distance'] * distance_score +
        weights['sequential'] * sequential_score +
        weights['hausdorff'] * hausdorff_score
    )
    
    return final_score
```

---

## Recommendation

**Recommended: Option 2 (Weighted Point-to-Edge Distance with Edge Coverage)**

**Rationale**:
1. **Good balance**: Combines proximity (distance) with actual edge matching (coverage)
2. **Efficient**: Reuses existing spatial index and candidate finding logic
3. **Robust**: Handles GPS inaccuracy well by considering multiple candidates
4. **Tunable**: Easy to adjust weights based on results
5. **Practical**: Not overly complex, but more sophisticated than simple distance

**Alternative**: If Option 2 doesn't perform well, consider **Option 6 (Hybrid)** for maximum robustness.

---

## Implementation Considerations

### Performance
- All options should be fast enough for 5 routes × ~10-50 GPS points
- Option 5 (Frechet) may be slowest - consider caching or approximation
- Reuse spatial index queries where possible

### Tuning Parameters
- **Distance scale factors**: 50-200 meters (depends on GPS accuracy)
- **Weights**: Start with equal weights, tune based on validation results
- **Lookahead window** (Option 3): 2-5 edges

### Edge Cases
- Empty routes: Return score = 0
- Single GPS point: Handle gracefully
- Very long routes: May need sampling or optimization
- GPS points far from all edges: Consider maximum distance threshold

---

## Next Steps

1. **Choose an option** (recommend Option 2)
2. **Implement similarity scoring function**
3. **Integrate into Step 1.3** to select route with highest score
4. **Update documentation** (SUMO_ROUTE_CALCULATION_LOGIC.md)
5. **Test and tune parameters** on sample data

