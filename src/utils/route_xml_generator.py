"""
Route XML generator for creating SUMO route files from parsed configurations.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PySide6.QtCore import QRectF
import random
import math


class RouteXMLGenerator:
    """Generator for SUMO route XML files."""
    
    def __init__(self, network_parser):
        """
        Initialize route XML generator.
        
        Args:
            network_parser: NetworkParser instance
        """
        self.network_parser = network_parser
        self.edges = network_parser.get_edges()
        self.nodes = network_parser.get_nodes()
    
    def generate_routes(
        self,
        config: Dict,
        source_areas: List[Tuple[QRectF, str]],
        target_areas: List[Tuple[QRectF, str]],
        output_path: str
    ):
        """
        Generate route XML file from configuration.
        
        Args:
            config: Parsed route configuration from LLM
            source_areas: List of (rect, id) tuples for source areas
            target_areas: List of (rect, id) tuples for target areas
            output_path: Path to output XML file
        """
        # Ensure config is a dict
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")
        
        # Create root element
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Add vehicle types
        self._add_vehicle_types(root)
        
        # Process route patterns - ensure it's a list
        route_patterns = config.get('route_patterns', [])
        if not isinstance(route_patterns, list):
            route_patterns = []
        
        special_events = config.get('special_events', [])
        if not isinstance(special_events, list):
            special_events = []
        
        vehicle_id_counter = 0
        
        # Generate routes for each pattern
        for pattern in route_patterns:
            vehicle_id_counter = self._generate_pattern_routes(
                root, pattern, source_areas, target_areas, vehicle_id_counter
            )
        
        # Generate routes for special events
        for event in special_events:
            vehicle_id_counter = self._generate_event_routes(
                root, event, source_areas, target_areas, vehicle_id_counter
            )
        
        # Write XML file
        self._write_xml(root, output_path)
    
    def _add_vehicle_types(self, root: ET.Element):
        """Add default vehicle types."""
        vtype_default = ET.SubElement(root, "vType")
        vtype_default.set("id", "DEFAULT_VEHTYPE")
        vtype_default.set("accel", "2.6")
        vtype_default.set("decel", "4.5")
        vtype_default.set("sigma", "0.5")
        vtype_default.set("length", "5.0")
        vtype_default.set("minGap", "2.5")
        vtype_default.set("maxSpeed", "70.0")
        vtype_default.set("color", "1,1,0")
    
    def _generate_pattern_routes(
        self,
        root: ET.Element,
        pattern: Dict,
        source_areas: List[Tuple[QRectF, str]],
        target_areas: List[Tuple[QRectF, str]],
        vehicle_id_counter: int
    ) -> int:
        """Generate routes for a route pattern."""
        # Ensure pattern is a dict
        if not isinstance(pattern, dict):
            return vehicle_id_counter
        
        sources = pattern.get('sources', [])
        if not isinstance(sources, list):
            sources = []
        
        targets = pattern.get('targets', [])
        if not isinstance(targets, list):
            targets = []
        
        vehicle_count = pattern.get('vehicle_count', 100)
        if not isinstance(vehicle_count, (int, float)):
            vehicle_count = 100
        vehicle_count = int(vehicle_count)
        time_period = pattern.get('time_period', {})
        
        # Parse time period - handle both dict and string formats
        if isinstance(time_period, dict):
            start_time = self._parse_time(time_period.get('start', '00:00'))
            end_time = self._parse_time(time_period.get('end', '23:59'))
        elif isinstance(time_period, str):
            # If time_period is a string (name), try to find it in config
            # For now, use default times
            start_time = self._parse_time('00:00')
            end_time = self._parse_time('23:59')
        else:
            # Default fallback
            start_time = self._parse_time('00:00')
            end_time = self._parse_time('23:59')
        
        duration = end_time - start_time
        if duration <= 0:
            duration = 3600  # Default 1 hour if invalid
        
        # Get source and target edges
        source_edges = self._get_edges_in_areas(sources, source_areas)
        target_edges = self._get_edges_in_areas(targets, target_areas)
        
        if not source_edges or not target_edges:
            return vehicle_id_counter
        
        # Generate vehicles
        for i in range(vehicle_count):
            # Random departure time within period
            dep_time = start_time + random.uniform(0, duration)
            
            # Random source and target edges
            source_edge = random.choice(source_edges)
            target_edge = random.choice(target_edges)
            
            # Find route between source and target
            route = self._find_route(source_edge, target_edge)
            
            if route:
                # Create vehicle
                vehicle = ET.SubElement(root, "vehicle")
                vehicle.set("id", f"veh_{vehicle_id_counter}")
                vehicle.set("depart", f"{dep_time:.1f}")
                vehicle.set("type", "DEFAULT_VEHTYPE")
                
                # Add route
                route_elem = ET.SubElement(vehicle, "route")
                route_elem.set("edges", " ".join(route))
                
                vehicle_id_counter += 1
        
        return vehicle_id_counter
    
    def _generate_event_routes(
        self,
        root: ET.Element,
        event: Dict,
        source_areas: List[Tuple[QRectF, str]],
        target_areas: List[Tuple[QRectF, str]],
        vehicle_id_counter: int
    ) -> int:
        """Generate routes for a special event."""
        # Ensure event is a dict
        if not isinstance(event, dict):
            return vehicle_id_counter
        
        sources = event.get('sources', [])
        if not isinstance(sources, list):
            sources = []
        
        targets = event.get('targets', [])
        if not isinstance(targets, list):
            targets = []
        
        vehicle_count = event.get('vehicle_count', 100)
        if not isinstance(vehicle_count, (int, float)):
            vehicle_count = 100
        vehicle_count = int(vehicle_count)
        
        event_time = self._parse_time(event.get('time', '12:00'))
        duration = event.get('duration', 3600)  # Default 1 hour
        if not isinstance(duration, (int, float)):
            duration = 3600
        duration = float(duration)
        
        # Get source and target edges
        source_edges = self._get_edges_in_areas(sources, source_areas)
        target_edges = self._get_edges_in_areas(targets, target_areas)
        
        if not source_edges or not target_edges:
            return vehicle_id_counter
        
        # Generate vehicles
        for i in range(vehicle_count):
            # Random departure time within event duration
            dep_time = event_time + random.uniform(0, duration)
            
            # Random source and target edges
            source_edge = random.choice(source_edges)
            target_edge = random.choice(target_edges)
            
            # Find route between source and target
            route = self._find_route(source_edge, target_edge)
            
            if route:
                # Create vehicle
                vehicle = ET.SubElement(root, "vehicle")
                vehicle.set("id", f"veh_{vehicle_id_counter}")
                vehicle.set("depart", f"{dep_time:.1f}")
                vehicle.set("type", "DEFAULT_VEHTYPE")
                
                # Add route
                route_elem = ET.SubElement(vehicle, "route")
                route_elem.set("edges", " ".join(route))
                
                vehicle_id_counter += 1
        
        return vehicle_id_counter
    
    def _get_edges_in_areas(self, area_refs: List[str], areas: List[Tuple[QRectF, str]]) -> List[str]:
        """Get edge IDs within specified areas."""
        edge_ids = []
        
        for area_ref in area_refs:
            # Parse area reference (e.g., "source_1" or "1")
            try:
                if area_ref.startswith('source_') or area_ref.startswith('target_'):
                    area_id = area_ref
                else:
                    # Assume numeric reference (1-based)
                    area_index = int(area_ref) - 1
                    if 0 <= area_index < len(areas):
                        area_id = areas[area_index][1]
                    else:
                        continue
            except (ValueError, IndexError):
                continue
            
            # Find area by ID
            area_rect = None
            for rect, aid in areas:
                if aid == area_id:
                    area_rect = rect
                    break
            
            if not area_rect:
                continue
            
            # Find edges within area
            for edge_id, edge_data in self.edges.items():
                # Check if any lane shape point is within area
                for lane in edge_data.get('lanes', []):
                    for point in lane.get('shape', []):
                        x, y = point
                        if (area_rect.x() <= x <= area_rect.x() + area_rect.width() and
                            area_rect.y() <= y <= area_rect.y() + area_rect.height()):
                            if edge_id not in edge_ids:
                                edge_ids.append(edge_id)
                            break
        
        return edge_ids
    
    def _find_route(self, source_edge: str, target_edge: str) -> Optional[List[str]]:
        """Find a route from source edge to target edge using simple BFS."""
        if source_edge == target_edge:
            return [source_edge]
        
        # Simple BFS to find route
        queue = [(source_edge, [source_edge])]
        visited = {source_edge}
        
        while queue:
            current_edge, path = queue.pop(0)
            
            # Get target node of current edge
            if current_edge not in self.edges:
                continue
            
            target_node = self.edges[current_edge]['to']
            
            # Find edges starting from target node
            for edge_id, edge_data in self.edges.items():
                if edge_data['from'] == target_node:
                    if edge_id == target_edge:
                        return path + [edge_id]
                    
                    if edge_id not in visited:
                        visited.add(edge_id)
                        queue.append((edge_id, path + [edge_id]))
        
        # If no route found, return direct path if possible
        return None
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string (HH:MM) to seconds."""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return hours * 3600 + minutes * 60
        except (ValueError, IndexError):
            return 0.0
    
    def _write_xml(self, root: ET.Element, output_path: str):
        """Write XML tree to file with pretty formatting."""
        # Convert to string
        rough_string = ET.tostring(root, encoding='unicode')
        
        # Parse with minidom for pretty printing
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

