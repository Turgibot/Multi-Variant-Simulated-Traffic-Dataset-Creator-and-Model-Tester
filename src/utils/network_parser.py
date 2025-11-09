"""
Network parser for extracting geometry from SUMO network files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
        
        self._parse()
    
    def _parse(self):
        """Parse the network file."""
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            
            # Parse location/bounds
            location = root.find('location')
            if location is not None:
                net_offset = location.get('netOffset', '0,0')
                conv_boundary = location.get('convBoundary', '0,0,0,0')
                orig_boundary = location.get('origBoundary', '0,0,0,0')
                
                # Parse boundaries
                try:
                    conv_parts = [float(x) for x in conv_boundary.split(',')]
                    if len(conv_parts) >= 4:
                        self.bounds = {
                            'x_min': conv_parts[0],
                            'y_min': conv_parts[1],
                            'x_max': conv_parts[2],
                            'y_max': conv_parts[3]
                        }
                except (ValueError, IndexError):
                    pass
            
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
                priority = int(edge.get('priority', '0'))
                
                # Parse lane shapes
                lanes = []
                for lane in edge.findall('lane'):
                    lane_id = lane.get('id')
                    shape_str = lane.get('shape', '')
                    width = float(lane.get('width', '3.2'))
                    speed = float(lane.get('speed', '13.89'))
                    
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
                        'speed': speed
                    })
                
                self.edges[edge_id] = {
                    'id': edge_id,
                    'from': from_node,
                    'to': to_node,
                    'priority': priority,
                    'lanes': lanes
                }
            
            # Parse junctions
            for junction in root.findall('junction'):
                junction_id = junction.get('id')
                x = float(junction.get('x', 0))
                y = float(junction.get('y', 0))
                junction_type = junction.get('type', '')
                
                self.junctions[junction_id] = {
                    'id': junction_id,
                    'x': x,
                    'y': y,
                    'type': junction_type
                }
            
            # If bounds not found, calculate from nodes/junctions
            if self.bounds is None:
                all_x = [n['x'] for n in self.nodes.values()] + [j['x'] for j in self.junctions.values()]
                all_y = [n['y'] for n in self.nodes.values()] + [j['y'] for j in self.junctions.values()]
                
                if all_x and all_y:
                    self.bounds = {
                        'x_min': min(all_x),
                        'y_min': min(all_y),
                        'x_max': max(all_x),
                        'y_max': max(all_y)
                    }
        
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

