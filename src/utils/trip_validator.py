"""
Trip validation utilities for GPS trajectory data.

These functions validate taxi trip trajectories and identify invalid segments
where the distance between consecutive GPS points exceeds reasonable thresholds.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple


class SegmentValidation(NamedTuple):
    """Validation result for a single segment between two GPS points."""
    start_index: int
    end_index: int
    distance_meters: float
    is_valid: bool


class TripValidationResult(NamedTuple):
    """Complete validation result for a trip."""
    is_valid: bool
    total_distance_meters: float
    segment_validations: List[SegmentValidation]
    invalid_segment_count: int
    invalid_segment_indices: List[int]  # Indices of start points of invalid segments


# Default maximum distance between consecutive GPS points (in meters)
DEFAULT_MAX_SEGMENT_DISTANCE = 1000.0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great-circle distance between two GPS points using the Haversine formula.
    
    Args:
        lon1: Longitude of first point (degrees)
        lat1: Latitude of first point (degrees)
        lon2: Longitude of second point (degrees)
        lat2: Latitude of second point (degrees)
    
    Returns:
        Distance in meters between the two points
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def validate_trip_segments(
    polyline: List[List[float]], 
    max_segment_distance: float = DEFAULT_MAX_SEGMENT_DISTANCE
) -> TripValidationResult:
    """
    Validate a trip by checking the distance between consecutive GPS points.
    
    A segment is considered invalid if the distance between two consecutive
    points exceeds the maximum allowed distance (default 500 meters).
    
    Args:
        polyline: List of [longitude, latitude] points representing the trip
        max_segment_distance: Maximum allowed distance between consecutive points (meters)
    
    Returns:
        TripValidationResult with detailed information about each segment
    
    Example:
        >>> polyline = [[-8.585676, 41.148522], [-8.585712, 41.148639], ...]
        >>> result = validate_trip_segments(polyline)
        >>> if not result.is_valid:
        ...     print(f"Found {result.invalid_segment_count} invalid segments")
        ...     for idx in result.invalid_segment_indices:
        ...         seg = result.segment_validations[idx]
        ...         print(f"  Segment {idx+1}-{idx+2}: {seg.distance_meters:.1f}m")
    """
    if not polyline or len(polyline) < 2:
        return TripValidationResult(
            is_valid=True,
            total_distance_meters=0.0,
            segment_validations=[],
            invalid_segment_count=0,
            invalid_segment_indices=[]
        )
    
    segment_validations = []
    invalid_indices = []
    total_distance = 0.0
    
    for i in range(len(polyline) - 1):
        lon1, lat1 = polyline[i]
        lon2, lat2 = polyline[i + 1]
        
        distance = haversine_distance(lon1, lat1, lon2, lat2)
        is_valid = distance <= max_segment_distance
        
        segment = SegmentValidation(
            start_index=i,
            end_index=i + 1,
            distance_meters=distance,
            is_valid=is_valid
        )
        segment_validations.append(segment)
        total_distance += distance
        
        if not is_valid:
            invalid_indices.append(i)
    
    return TripValidationResult(
        is_valid=len(invalid_indices) == 0,
        total_distance_meters=total_distance,
        segment_validations=segment_validations,
        invalid_segment_count=len(invalid_indices),
        invalid_segment_indices=invalid_indices
    )


def get_segment_distances(polyline: List[List[float]]) -> List[float]:
    """
    Calculate distances between all consecutive GPS points in a polyline.
    
    Args:
        polyline: List of [longitude, latitude] points
    
    Returns:
        List of distances in meters between consecutive points
    """
    if not polyline or len(polyline) < 2:
        return []
    
    distances = []
    for i in range(len(polyline) - 1):
        lon1, lat1 = polyline[i]
        lon2, lat2 = polyline[i + 1]
        distances.append(haversine_distance(lon1, lat1, lon2, lat2))
    
    return distances


def get_trip_statistics(polyline: List[List[float]]) -> Dict:
    """
    Calculate statistics for a trip's GPS trajectory.
    
    Args:
        polyline: List of [longitude, latitude] points
    
    Returns:
        Dictionary with statistics including total distance, average segment length,
        max segment length, and more
    """
    if not polyline:
        return {
            'point_count': 0,
            'segment_count': 0,
            'total_distance_meters': 0.0,
            'avg_segment_distance': 0.0,
            'min_segment_distance': 0.0,
            'max_segment_distance': 0.0,
        }
    
    distances = get_segment_distances(polyline)
    
    if not distances:
        return {
            'point_count': len(polyline),
            'segment_count': 0,
            'total_distance_meters': 0.0,
            'avg_segment_distance': 0.0,
            'min_segment_distance': 0.0,
            'max_segment_distance': 0.0,
        }
    
    return {
        'point_count': len(polyline),
        'segment_count': len(distances),
        'total_distance_meters': sum(distances),
        'avg_segment_distance': sum(distances) / len(distances),
        'min_segment_distance': min(distances),
        'max_segment_distance': max(distances),
    }





