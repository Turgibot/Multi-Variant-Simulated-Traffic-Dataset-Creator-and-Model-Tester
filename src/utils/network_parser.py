"""
Network parser for extracting geometry from SUMO network files.
"""

import gzip
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import pyproj for accurate coordinate conversion
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


class NetworkParser:
    """Parser for SUMO network files (.net.xml)."""
    
    def __init__(self, net_file: str):
        """
        Initialize network parser.
        
        Args:
            net_file: Path to the .net.xml file
        """
        self.net_file = Path(net_file)
        if not self.net_file.exists():
            raise ValueError(f"Network file does not exist: {net_file}")
        
        self.nodes = {}
        self.edges = {}
        self.junctions = {}
        self.bounds = None
        self.orig_boundary = None  # Original GPS boundary for coordinate conversion
        self.net_offset = None  # Network offset
        self.proj_parameter = None  # Projection parameters from network file
        self.transformer = None  # PyProj transformer if available
        
        # Adjustment factors for fine-tuning coordinate conversion
        self.conv_adjust_x = 0.0  # Adjustment to normalized X position (-1 to 1)
        self.conv_adjust_y = 0.0  # Adjustment to normalized Y position (-1 to 1)
        self.conv_scale_x = 1.0   # Scale factor for X dimension
        self.conv_scale_y = 1.0   # Scale factor for Y dimension
        
        self._parse()
    
    def _parse(self):
        """Parse the network file."""
        try:
            # Handle compressed files (.gz)
            if self.net_file.suffix == '.gz' or self.net_file.name.endswith('.gz'):
                # Open compressed file
                with gzip.open(self.net_file, 'rb') as f:
                    tree = ET.parse(f)
            else:
                # Open regular file
                tree = ET.parse(self.net_file)
            
            root = tree.getroot()
            
            # Parse location/bounds
            location = root.find('location')
            if location is not None:
                net_offset_str = location.get('netOffset', '0,0')
                conv_boundary = location.get('convBoundary', '0,0,0,0')
                orig_boundary_str = location.get('origBoundary', '0,0,0,0')
                self.proj_parameter = location.get('projParameter', '')
                
                # Parse net offset
                try:
                    offset_parts = [float(x) for x in net_offset_str.split(',')]
                    if len(offset_parts) >= 2:
                        self.net_offset = {'x': offset_parts[0], 'y': offset_parts[1]}
                except (ValueError, IndexError):
                    pass
                
                # Parse original boundary (GPS coordinates)
                try:
                    orig_parts = [float(x) for x in orig_boundary_str.split(',')]
                    if len(orig_parts) >= 4:
                        self.orig_boundary = {
                            'lon_min': orig_parts[0],
                            'lat_min': orig_parts[1],
                            'lon_max': orig_parts[2],
                            'lat_max': orig_parts[3]
                        }
                except (ValueError, IndexError):
                    pass
                
                # Initialize coordinate transformer if pyproj is available
                # Now using improved method: pyproj for accurate projection, then map to convBoundary
                if PYPROJ_AVAILABLE and self.proj_parameter:
                    try:
                        # Create transformer from WGS84 (GPS) to network projection
                        wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (GPS coordinates)
                        network_proj = pyproj.CRS.from_string(self.proj_parameter)
                        self.transformer = pyproj.Transformer.from_crs(wgs84, network_proj, always_xy=True)
                    except Exception as e:
                        print(f"⚠️ Failed to initialize pyproj transformer: {e}")
                        print(f"   Proj parameter: {self.proj_parameter}")
                        print(f"   Falling back to linear interpolation")
                        self.transformer = None
                else:
                    self.transformer = None
                    if not PYPROJ_AVAILABLE:
                        print("⚠️ Pyproj not available, using linear interpolation")
                    elif not self.proj_parameter:
                        print("⚠️ Network file missing projParameter, using linear interpolation")
                
                # Parse converted boundary (SUMO coordinates)
                # Store original convBoundary separately - needed for GPS-to-SUMO conversion
                try:
                    conv_parts = [float(x) for x in conv_boundary.split(',')]
                    if len(conv_parts) >= 4:
                        self.conv_boundary = {
                            'x_min': conv_parts[0],
                            'y_min': conv_parts[1],
                            'x_max': conv_parts[2],
                            'y_max': conv_parts[3]
                        }
                        # Also set bounds initially (will be recalculated from actual edges later)
                        self.bounds = {
                            'x_min': conv_parts[0],
                            'y_min': conv_parts[1],
                            'x_max': conv_parts[2],
                            'y_max': conv_parts[3]
                        }
                except (ValueError, IndexError):
                    self.conv_boundary = None
            
            # Parse nodes
            for node in root.findall('node'):
                node_id = node.get('id')
                x = float(node.get('x', 0))
                y = float(node.get('y', 0))
                node_type = node.get('type', '')
                
                self.nodes[node_id] = {
                    'id': node_id,
                    'x': x,
                    'y': y,
                    'type': node_type
                }
            
            # Parse edges
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                from_node = edge.get('from')
                to_node = edge.get('to')
                priority_str = edge.get('priority')
                
                # DO NOT skip internal edges - load everything from XML
                # Internal edges (with ':' in ID) are still valid edges in the network
                # if ':' in edge_id:
                #     continue
                
                # Skip edges without required attributes (from, to, priority)
                if not from_node or not to_node or priority_str is None:
                    continue
                
                try:
                    priority = int(priority_str)
                except (ValueError, TypeError):
                    continue
                
                # Parse lane shapes and vehicle class restrictions
                lanes = []
                edge_has_restrictions = False  # Track if any lane has 'allow' restrictions
                edge_allows_passenger = False  # Track if any lane allows passenger vehicles
                for lane in edge.findall('lane'):
                    lane_id = lane.get('id')
                    shape_str = lane.get('shape', '')
                    width = float(lane.get('width', '3.2'))
                    speed = float(lane.get('speed', '13.89'))
                    length_str = lane.get('length', None)  # Lane length in meters
                    allow_str = lane.get('allow', '')  # Vehicle classes allowed on this lane
                    
                    # Parse lane length
                    lane_length = None
                    if length_str:
                        try:
                            lane_length = float(length_str)
                        except (ValueError, TypeError):
                            pass
                    
                    # Parse allowed vehicle classes
                    if allow_str:
                        edge_has_restrictions = True
                        # SUMO allows multiple vehicle classes separated by spaces
                        allowed_classes = allow_str.split()
                        # Check if this lane allows passenger/taxi/private vehicles
                        if ('passenger' in allowed_classes or 
                            'taxi' in allowed_classes or 
                            'private' in allowed_classes):
                            edge_allows_passenger = True
                    
                    # Parse shape coordinates
                    shape_points = []
                    if shape_str:
                        coords = shape_str.split()
                        for coord in coords:
                            try:
                                x, y = map(float, coord.split(','))
                                shape_points.append((x, y))
                            except (ValueError, IndexError):
                                continue
                    
                    lanes.append({
                        'id': lane_id,
                        'shape': shape_points,
                        'width': width,
                        'speed': speed,
                        'length': lane_length,
                        'allow': allow_str
                    })
                
                # Determine if edge allows passenger vehicles:
                # - If no restrictions (no 'allow' attributes), all vehicles allowed -> True
                # - If restrictions exist and passenger/taxi is allowed -> True
                # - If restrictions exist but passenger/taxi not allowed -> False
                allows_passenger = True if not edge_has_restrictions else edge_allows_passenger
                
                self.edges[edge_id] = {
                    'id': edge_id,
                    'from': from_node,
                    'to': to_node,
                    'priority': priority,
                    'lanes': lanes,
                    'allows_passenger': allows_passenger  # True/False/None (None = all allowed)
                }
            
            # Parse junctions
            for junction in root.findall('junction'):
                junction_id = junction.get('id')
                x = float(junction.get('x', 0))
                y = float(junction.get('y', 0))
                junction_type = junction.get('type', '')
                shape_str = junction.get('shape', '')
                
                # Parse junction shape if available
                shape_points = []
                if shape_str:
                    coords = shape_str.split()
                    for coord in coords:
                        try:
                            x_coord, y_coord = map(float, coord.split(','))
                            shape_points.append((x_coord, y_coord))
                        except (ValueError, IndexError):
                            continue
                
                self.junctions[junction_id] = {
                    'id': junction_id,
                    'x': x,
                    'y': y,
                    'type': junction_type,
                    'shape': shape_points  # Store actual junction shape polygon
                }
            
            # Calculate actual bounds from all network elements (edges, nodes, junctions)
            # This ensures bounds match what's actually drawn, not just convBoundary
            all_x = []
            all_y = []
            
            # Add node coordinates
            for n in self.nodes.values():
                all_x.append(n['x'])
                all_y.append(n['y'])
            
            # Add junction coordinates
            for j in self.junctions.values():
                all_x.append(j['x'])
                all_y.append(j['y'])
            
            # Add edge shape coordinates (most important - this is what's actually drawn)
            for edge_data in self.edges.values():
                lanes = edge_data.get('lanes', [])
                for lane in lanes:
                    shape = lane.get('shape', [])
                    for x, y in shape:
                        all_x.append(x)
                        all_y.append(y)
            
            # Calculate actual bounds from all coordinates
            if all_x and all_y:
                actual_bounds = {
                    'x_min': min(all_x),
                    'y_min': min(all_y),
                    'x_max': max(all_x),
                    'y_max': max(all_y)
                }
                
                # Use actual bounds instead of convBoundary for better alignment
                # convBoundary might include padding that the actual network doesn't use
                self.bounds = actual_bounds
        
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse network file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading network file: {e}")
    
    def get_bounds(self) -> Optional[Dict[str, float]]:
        """Get network bounds."""
        return self.bounds
    
    def get_nodes(self) -> Dict:
        """Get all nodes."""
        return self.nodes
    
    def get_edges(self) -> Dict:
        """Get all edges."""
        return self.edges
    
    def get_junctions(self) -> Dict:
        """Get all junctions."""
        return self.junctions
    
    def gps_to_sumo_coords(self, lon: float, lat: float, force_linear: bool = False) -> Optional[Tuple[float, float]]:
        """
        Convert GPS coordinates (longitude, latitude) to SUMO network coordinates (x, y).
        
        SUMO networks use a projected coordinate system. The conversion works as follows:
        - origBoundary: GPS bounds (lon, lat) of the original area
        - convBoundary: Projected bounds (x, y) relative to net_offset
        - net_offset: Origin point in the projected coordinate system
        
        Network coordinates = projected coordinates - net_offset
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Tuple of (x, y) in SUMO coordinates, or None if conversion not possible
        """
        # Use improved projection-aware conversion
        # SUMO's convBoundary is in its internal coordinate system, not raw projected coordinates
        # So we use pyproj to get accurate relative positions, then map to convBoundary
        if self.transformer is not None and not force_linear and self.orig_boundary and self.conv_boundary:
            try:
                # Transform GPS (WGS84) to network projection coordinates
                proj_x, proj_y = self.transformer.transform(lon, lat)
                
                # Validate transformed coordinates
                if not (math.isfinite(proj_x) and math.isfinite(proj_y)):
                    raise ValueError(f"Invalid coordinates from transformer: ({proj_x}, {proj_y})")
                
                # Transform the corners of origBoundary to get the projected bounds
                # This gives us the actual projected coordinate range
                sw_lon, sw_lat = self.orig_boundary['lon_min'], self.orig_boundary['lat_min']
                ne_lon, ne_lat = self.orig_boundary['lon_max'], self.orig_boundary['lat_max']
                
                sw_proj_x, sw_proj_y = self.transformer.transform(sw_lon, sw_lat)
                ne_proj_x, ne_proj_y = self.transformer.transform(ne_lon, ne_lat)
                
                # Calculate normalized position within projected bounds
                proj_x_min = min(sw_proj_x, ne_proj_x)
                proj_x_max = max(sw_proj_x, ne_proj_x)
                proj_y_min = min(sw_proj_y, ne_proj_y)
                proj_y_max = max(sw_proj_y, ne_proj_y)
                
                if proj_x_max == proj_x_min or proj_y_max == proj_y_min:
                    raise ValueError("Invalid projected bounds")
                
                # Normalize position within projected bounds
                proj_x_norm = (proj_x - proj_x_min) / (proj_x_max - proj_x_min)
                proj_y_norm = (proj_y - proj_y_min) / (proj_y_max - proj_y_min)
                
                # Apply adjustment factors (scale and offset to normalized position)
                proj_x_norm = (proj_x_norm * self.conv_scale_x) + self.conv_adjust_x
                proj_y_norm = (proj_y_norm * self.conv_scale_y) + self.conv_adjust_y
                
                # Clamp to valid range [0, 1] after adjustments
                proj_x_norm = max(0.0, min(1.0, proj_x_norm))
                proj_y_norm = max(0.0, min(1.0, proj_y_norm))
                
                # Map normalized position to convBoundary (SUMO's internal coordinate system)
                x = self.conv_boundary['x_min'] + proj_x_norm * (self.conv_boundary['x_max'] - self.conv_boundary['x_min'])
                y = self.conv_boundary['y_min'] + proj_y_norm * (self.conv_boundary['y_max'] - self.conv_boundary['y_min'])
                
                # Validate final coordinates
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"Invalid coordinates after mapping: ({x}, {y})")
                
                return (x, y)
            except Exception as e:
                # If transformation fails, fall back to linear interpolation
                # #region agent log
                try:
                    with open('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({"sessionId":"debug-session","runId":"proj-fix-v2","hypothesisId":"D","location":"network_parser.py:gps_to_sumo_coords","message":"Pyproj transformation failed, using fallback","data":{"error":str(e),"gps":(lon,lat)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                except:
                    pass
                # #endregion
                pass
        
        # Fallback: Use linear interpolation (less accurate, especially at edges)
        # Use conv_boundary (original convBoundary) for conversion, not bounds (which are from actual edges)
        if not self.orig_boundary or not self.conv_boundary:
            return None
        
        # Use linear interpolation: map GPS coordinates (origBoundary) directly to network coordinates (convBoundary)
        # convBoundary is already in network coordinates, so we map directly without any offset
        lon_min = self.orig_boundary['lon_min']
        lon_max = self.orig_boundary['lon_max']
        lat_min = self.orig_boundary['lat_min']
        lat_max = self.orig_boundary['lat_max']
        
        # convBoundary represents network coordinates directly - use original convBoundary, not recalculated bounds
        net_x_min = self.conv_boundary['x_min']
        net_x_max = self.conv_boundary['x_max']
        net_y_min = self.conv_boundary['y_min']
        net_y_max = self.conv_boundary['y_max']
        
        # Check for valid boundaries
        if lon_max == lon_min or lat_max == lat_min:
            return None
        
        # Normalize GPS coordinates to [0, 1] within origBoundary
        lon_norm = (lon - lon_min) / (lon_max - lon_min)
        lat_norm = (lat - lat_min) / (lat_max - lat_min)
        
        # Apply adjustment factors (scale and offset to normalized position)
        lon_norm = (lon_norm * self.conv_scale_x) + self.conv_adjust_x
        lat_norm = (lat_norm * self.conv_scale_y) + self.conv_adjust_y
        
        # Clamp to valid range [0, 1] after adjustments
        lon_norm = max(0.0, min(1.0, lon_norm))
        lat_norm = max(0.0, min(1.0, lat_norm))
        
        # Map normalized position directly to network coordinates (convBoundary)
        x = net_x_min + lon_norm * (net_x_max - net_x_min)
        y = net_y_min + lat_norm * (net_y_max - net_y_min)
        
        # #region agent log
        with open('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"proj-fix","hypothesisId":"C","location":"network_parser.py:gps_to_sumo_coords","message":"Using linear interpolation fallback","data":{"gps":(lon,lat),"sumo":(x,y),"orig_boundary":self.orig_boundary,"conv_boundary":self.conv_boundary,"bounds":self.bounds,"lon_norm":lon_norm,"lat_norm":lat_norm,"method":"linear","transformer_available":self.transformer is not None,"proj_parameter":self.proj_parameter},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        return (x, y)

    def sumo_to_gps_coords(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Convert SUMO network coordinates to GPS (lon, lat).
        Inverse of gps_to_sumo_coords when using linear interpolation fallback.

        Args:
            x: X coordinate in SUMO space
            y: Y coordinate in SUMO space

        Returns:
            Tuple of (lon, lat) or None if conversion fails
        """
        if not self.orig_boundary or not self.conv_boundary:
            return None

        lon_min = self.orig_boundary['lon_min']
        lon_max = self.orig_boundary['lon_max']
        lat_min = self.orig_boundary['lat_min']
        lat_max = self.orig_boundary['lat_max']

        net_x_min = self.conv_boundary['x_min']
        net_x_max = self.conv_boundary['x_max']
        net_y_min = self.conv_boundary['y_min']
        net_y_max = self.conv_boundary['y_max']

        if net_x_max == net_x_min or net_y_max == net_y_min:
            return None

        x_norm = (x - net_x_min) / (net_x_max - net_x_min)
        y_norm = (y - net_y_min) / (net_y_max - net_y_min)
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        lon_norm = (x_norm - self.conv_adjust_x) / self.conv_scale_x
        lat_norm = (y_norm - self.conv_adjust_y) / self.conv_scale_y
        lon_norm = max(0.0, min(1.0, lon_norm))
        lat_norm = max(0.0, min(1.0, lat_norm))

        lon = lon_min + lon_norm * (lon_max - lon_min)
        lat = lat_min + lat_norm * (lat_max - lat_min)

        return (lon, lat)
    
    def _point_to_segment_distance_and_position(self, px: float, py: float, 
                                                 x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """
        Calculate distance from point to line segment and position along segment.
        
        Args:
            px, py: Point coordinates
            x1, y1: Segment start coordinates
            x2, y2: Segment end coordinates
            
        Returns:
            Tuple of (distance from point to segment, position along segment from start)
        """
        # Vector from segment start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Length of segment squared
        len_sq = dx * dx + dy * dy
        
        if len_sq == 0:
            # Segment is a point
            dist = math.sqrt((px - x1)**2 + (py - y1)**2)
            return (dist, 0.0)
        
        # Project point onto line (find parameter t where closest point is at start + t * (end - start))
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on segment
        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        
        # Position along segment (distance from start to closest point)
        pos = t * math.sqrt(len_sq)
        
        return (dist, pos)
    
    def find_nearest_edge(self, x: float, y: float, allow_passenger_only: bool = False) -> Optional[Tuple[str, float, float]]:
        """
        Find the nearest edge to a given SUMO coordinate point.
        
        Args:
            x: X coordinate in SUMO space
            y: Y coordinate in SUMO space
            allow_passenger_only: If True, only return edges that allow passenger/taxi vehicles
            
        Returns:
            Tuple of (edge_id, distance, position_along_edge) or None if no edge found
            position_along_edge is the distance in meters from the start of the edge
        """
        min_distance = float('inf')
        nearest_edge_id = None
        nearest_position = 0.0
        
        for edge_id, edge_data in self.edges.items():
            # Filter by vehicle class if requested
            if allow_passenger_only:
                allows_passenger = edge_data.get('allows_passenger')
                # Skip edges that explicitly don't allow passenger vehicles
                if allows_passenger is False:
                    continue
            
            # Check all lanes (use first lane as representative of the edge)
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            # Use first lane's shape
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Check each segment of the edge
            accumulated_length = 0.0
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                
                dist, pos_on_segment = self._point_to_segment_distance_and_position(x, y, x1, y1, x2, y2)
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_edge_id = edge_id
                    nearest_position = accumulated_length + pos_on_segment
                
                # Add segment length to accumulated length
                accumulated_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if nearest_edge_id:
            # Clamp position to valid range [0, edge_length]
            # SUMO requires stop positions to be within edge bounds
            edge_length = self.get_edge_length(nearest_edge_id)
            if edge_length > 0:
                # Keep a safe margin (1.0m) from the end to avoid edge boundary issues
                max_pos = max(0.0, edge_length - 1.0)
                nearest_position = max(0.0, min(nearest_position, max_pos))
            else:
                nearest_position = 0.0
            
            return (nearest_edge_id, min_distance, nearest_position)
        return None
    
    def get_edge_length(self, edge_id: str) -> float:
        """
        Get the length of an edge using the lane's length attribute.
        Falls back to calculating from shape if length attribute not available.
        
        Args:
            edge_id: The edge ID
            
        Returns:
            Length in meters, or 0 if edge not found
        """
        if edge_id not in self.edges:
            return 0.0
        
        edge_data = self.edges[edge_id]
        lanes = edge_data.get('lanes', [])
        if not lanes:
            return 0.0
        
        # First, try to use the lane's length attribute (most accurate)
        lane_length = lanes[0].get('length')
        if lane_length is not None and lane_length > 0:
            return lane_length
        
        # Fall back to calculating from shape
        shape = lanes[0].get('shape', [])
        if len(shape) < 2:
            return 0.0
        
        # Calculate total length
        total_length = 0.0
        for i in range(len(shape) - 1):
            x1, y1 = shape[i]
            x2, y2 = shape[i + 1]
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return total_length
    
    def get_first_lane_id(self, edge_id: str) -> Optional[str]:
        """
        Get the ID of the first lane on an edge.
        
        Args:
            edge_id: The edge ID
            
        Returns:
            Lane ID string, or None if edge not found or has no lanes
        """
        if edge_id not in self.edges:
            return None
        
        edge_data = self.edges[edge_id]
        lanes = edge_data.get('lanes', [])
        if not lanes:
            return None
        
        return lanes[0].get('id')
    
    def edge_allows_passenger(self, edge_id: str) -> bool:
        """
        Check if an edge allows passenger/taxi vehicles.
        
        Args:
            edge_id: The edge ID to check
            
        Returns:
            True if edge allows passenger vehicles, False otherwise
        """
        if edge_id not in self.edges:
            return False
        
        edge_data = self.edges[edge_id]
        allows_passenger = edge_data.get('allows_passenger', True)  # Default to True if not set
        
        # Boolean value: True = allows passenger, False = doesn't allow
        return allows_passenger

