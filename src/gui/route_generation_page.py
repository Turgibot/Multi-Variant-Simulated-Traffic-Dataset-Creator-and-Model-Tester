"""
Route Generation page for creating SUMO route files manually.
"""

import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QGraphicsView, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.route_xml_generator import RouteXMLGenerator
from src.utils.sumo_config_manager import SUMOConfigManager


class AreaSelectionView(SimulationView):
    """Extended SimulationView with area selection capabilities."""
    
    area_selected = Signal(str, QRectF)  # Emits area_type, rectangle
    zone_area_selected = Signal(str, QRectF)  # Emits zone_id, rectangle
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_mode = None  # 'source', 'target', 'zone', or None
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.source_areas = []  # List of (rect, id) tuples
        self.target_areas = []  # List of (rect, id) tuples
        self.zones = {}  # Dict of {zone_id: {'areas': [QRectF], 'name': str, 'color': QColor}}
        self.area_counter = 0
        self.current_zone_id = None  # Zone ID being selected
        
        # Override drag mode for selection
        self.setDragMode(QGraphicsView.NoDrag)
    
    def set_selection_mode(self, mode: Optional[str], zone_id: Optional[str] = None):
        """Set selection mode: 'source', 'target', 'zone', or None."""
        self.selection_mode = mode
        self.current_zone_id = zone_id if mode == 'zone' else None
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)
    
    def mousePressEvent(self, event):
        """Handle mouse press for area selection."""
        if self.selection_mode and event.button() == Qt.LeftButton:
            self.selecting = True
            scene_pos = self.mapToScene(event.pos())
            self.selection_start = scene_pos
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for area selection."""
        if self.selecting and self.selection_start:
            scene_pos = self.mapToScene(event.pos())
            # Create selection rectangle
            self.selection_rect = QRectF(
                min(self.selection_start.x(), scene_pos.x()),
                min(self.selection_start.y(), scene_pos.y()),
                abs(scene_pos.x() - self.selection_start.x()),
                abs(scene_pos.y() - self.selection_start.y())
            )
            self.viewport().update()  # Trigger repaint
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for area selection."""
        if self.selecting and self.selection_rect and event.button() == Qt.LeftButton:
            if self.selection_mode == 'zone' and self.current_zone_id:
                # Add zone area (allow multiple areas per zone)
                if self.current_zone_id in self.zones:
                    if 'areas' not in self.zones[self.current_zone_id]:
                        self.zones[self.current_zone_id]['areas'] = []
                    self.zones[self.current_zone_id]['areas'].append(self.selection_rect)
                # Emit signal
                self.zone_area_selected.emit(self.current_zone_id, self.selection_rect)
                # Keep selection mode active for more areas
                self.selecting = False
                self.selection_start = None
                self.selection_rect = None
                self.viewport().update()
            else:
                # Add area to appropriate list
                self.area_counter += 1
                area_id = f"{self.selection_mode}_{self.area_counter}"
                
                if self.selection_mode == 'source':
                    self.source_areas.append((self.selection_rect, area_id))
                elif self.selection_mode == 'target':
                    self.target_areas.append((self.selection_rect, area_id))
                
                # Emit signal
                self.area_selected.emit(self.selection_mode, self.selection_rect)
                
                # Reset selection
                self.selecting = False
                self.selection_start = None
                self.selection_rect = None
                self.selection_mode = None
                self.current_zone_id = None
                self.setCursor(Qt.ArrowCursor)
                self.viewport().update()
        else:
            super().mouseReleaseEvent(event)
    
    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Override to draw selection rectangle and areas."""
        super().drawForeground(painter, rect)
        
        # Get zones_locked from the route generation page
        zones_locked = False
        if hasattr(self, 'route_page'):
            zones_locked = self.route_page.zones_locked
        
        # Draw all zones with their colors and names
        for zone_id, zone_data in self.zones.items():
            zone_color = zone_data.get('color', QColor(100, 100, 100))
            zone_name = zone_data.get('name', '')
            zone_areas = zone_data.get('areas', [])
            
            if zones_locked:
                # Locked mode: show only zone name in center of largest area
                if zone_areas and zone_name:
                    # Find largest area
                    largest_area = max(zone_areas, key=lambda r: r.width() * r.height())
                    center_x = largest_area.x() + largest_area.width() / 2
                    center_y = largest_area.y() + largest_area.height() / 2
                    
                    # Draw zone name in center
                    painter.setPen(QPen(zone_color, 2))
                    painter.setBrush(QBrush(QColor(255, 255, 255, 220)))  # Semi-transparent white background
                    font = QFont("Arial", 14, QFont.Bold)
                    painter.setFont(font)
                    # Get text metrics
                    metrics = painter.fontMetrics()
                    text_rect = metrics.boundingRect(zone_name)
                    # Center the text
                    text_x = center_x - text_rect.width() / 2
                    text_y = center_y + text_rect.height() / 2
                    bg_rect = QRectF(text_x - 5, text_y - text_rect.height() - 3, 
                                    text_rect.width() + 10, text_rect.height() + 6)
                    painter.drawRect(bg_rect)
                    # Draw text
                    painter.setPen(QPen(zone_color, 2))
                    painter.drawText(text_x, text_y, zone_name)
            else:
                # Normal mode: draw colored areas with names
                # Draw all areas for this zone
                for zone_rect in zone_areas:
                    # Draw zone area
                    painter.setPen(QPen(zone_color, 2))
                    painter.setBrush(QBrush(QColor(zone_color.red(), zone_color.green(), zone_color.blue(), 30)))
                    painter.drawRect(zone_rect)
                    
                    # Draw zone name at top corner of each area
                    if zone_name:
                        # Draw background rectangle for text
                        painter.setPen(QPen(zone_color, 1))
                        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))  # Semi-transparent white background
                        font = QFont("Arial", 12, QFont.Bold)
                        painter.setFont(font)
                        # Get text metrics
                        metrics = painter.fontMetrics()
                        text_rect = metrics.boundingRect(zone_name)
                        # Position at top-left corner with small offset
                        text_x = zone_rect.x() + 5
                        text_y = zone_rect.y() + text_rect.height() + 5
                        bg_rect = QRectF(text_x - 2, text_y - text_rect.height() - 2, 
                                        text_rect.width() + 4, text_rect.height() + 4)
                        painter.drawRect(bg_rect)
                        # Draw text
                        painter.setPen(QPen(zone_color, 2))
                        painter.drawText(text_x, text_y, zone_name)
        
        # Draw all selected source areas
        for rect_item, area_id in self.source_areas:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
            painter.drawRect(rect_item)
        
        # Draw all selected target areas
        for rect_item, area_id in self.target_areas:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0, 30)))
            painter.drawRect(rect_item)
        
        # Draw current selection rectangle
        if self.selecting and self.selection_rect:
            if self.selection_mode == 'zone':
                # Use zone color if available
                if self.current_zone_id and self.current_zone_id in self.zones:
                    zone_color = self.zones[self.current_zone_id].get('color', QColor(0, 150, 255))
                else:
                    zone_color = QColor(0, 150, 255)
                painter.setPen(QPen(zone_color, 2, Qt.DashLine))
                painter.setBrush(QBrush(QColor(zone_color.red(), zone_color.green(), zone_color.blue(), 30)))
            else:
                painter.setPen(QPen(QColor(0, 150, 255), 2, Qt.DashLine))
                painter.setBrush(QBrush(QColor(0, 150, 255, 30)))
            painter.drawRect(self.selection_rect)
    
    def clear_areas(self, area_type: Optional[str] = None):
        """Clear selected areas. If area_type is None, clear all."""
        if area_type == 'source':
            self.source_areas.clear()
        elif area_type == 'target':
            self.target_areas.clear()
        else:
            self.source_areas.clear()
            self.target_areas.clear()
        self.viewport().update()
    
    def get_areas(self, area_type: str) -> List[Tuple[QRectF, str]]:
        """Get areas of specified type."""
        if area_type == 'source':
            return self.source_areas.copy()
        elif area_type == 'target':
            return self.target_areas.copy()
        return []


class RouteGenerationPage(QWidget):
    """Page for generating SUMO route files manually."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.config_manager = SUMOConfigManager(project_path)
        self.network_parser = None
        self.route_generator = None
        self.zones = {}  # Dict of {zone_id: {'name': str, 'color': QColor, 'widget': QWidget}}
        self.zone_counter = 0  # Counter for zone letters (A, B, C, ...)
        self.current_zone_id = None  # Currently active zone for selection
        self.zones_locked = False  # Whether zones are locked (done button clicked)
        self.displayed_zones = set()  # Set of zone IDs that are currently being displayed (highlighted)
        
        self.init_ui()
        self.load_network()
    
    def init_ui(self):
        """Initialize the page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel(f"Route Generation - {self.project_name}")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        header_layout.addWidget(back_btn)
        main_layout.addLayout(header_layout)
        
        # Main content: split into map and configuration
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Left side: Map view
        map_group = QGroupBox("Network Map")
        map_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        map_layout = QVBoxLayout()
        map_layout.setSpacing(10)
        
        self.map_view = AreaSelectionView()
        self.map_view.setMinimumSize(600, 500)
        self.map_view.area_selected.connect(self.on_area_selected)
        self.map_view.zone_area_selected.connect(self.on_zone_area_selected)
        # Store reference to parent for zones_locked access
        self.map_view.route_page = self
        map_layout.addWidget(self.map_view)
        map_group.setLayout(map_layout)
        content_layout.addWidget(map_group, stretch=2)
        
        # Right side: Route Configuration
        config_group = QGroupBox("Route Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        config_layout = QVBoxLayout()
        config_layout.setSpacing(15)
        config_layout.setContentsMargins(10, 10, 10, 10)
        
        # Zones section
        zones_header = QHBoxLayout()
        zones_header.setSpacing(10)
        
        zones_title = QLabel("Zones")
        zones_title.setFont(QFont("Arial", 14, QFont.Bold))
        zones_title.setStyleSheet("color: #333;")
        zones_header.addWidget(zones_title)
        
        add_zone_btn = QPushButton("+")
        add_zone_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
                font-size: 18px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_zone_btn.clicked.connect(self.add_zone)
        zones_header.addWidget(add_zone_btn)
        
        zones_header.addStretch()
        
        # Done button
        self.done_btn = QPushButton("Done")
        self.done_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.done_btn.clicked.connect(self.lock_zones)
        zones_header.addWidget(self.done_btn)
        
        config_layout.addLayout(zones_header)
        
        # Zones list/container (empty for now)
        self.zones_container = QVBoxLayout()
        self.zones_container.setSpacing(10)
        config_layout.addLayout(self.zones_container)
        
        config_layout.addStretch()
        config_group.setLayout(config_layout)
        content_layout.addWidget(config_group, stretch=1)
        
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
    
    def load_network(self):
        """Load network from sumocfg if available."""
        sumocfg_path = self.config_manager.get_sumocfg_path()
        if not sumocfg_path:
            return
        
        try:
            import xml.etree.ElementTree as ET
            sumocfg_path = Path(sumocfg_path)
            if not sumocfg_path.exists():
                return
            
            tree = ET.parse(sumocfg_path)
            root = tree.getroot()
            input_elem = root.find('input')
            
            if input_elem is not None:
                net_elem = input_elem.find('net-file')
                if net_elem is not None:
                    net_file = net_elem.get('value')
                else:
                    net_file = input_elem.get('net-file')
                
                if net_file:
                    net_path = (sumocfg_path.parent / net_file).resolve()
                    if net_path.exists():
                        self.network_parser = NetworkParser(str(net_path))
                        self.map_view.load_network(self.network_parser)
                        self.route_generator = RouteXMLGenerator(self.network_parser)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load network: {str(e)}")
    
    def on_area_selected(self, area_type: str, rect: QRectF):
        """Handle area selection."""
        # Placeholder for future implementation
        pass
    
    def on_zone_area_selected(self, zone_id: str, rect: QRectF):
        """Handle zone area selection."""
        if zone_id in self.zones:
            # Update zone in map view (already done in mouseReleaseEvent)
            # Update counts in widget and visualization
            self.update_zone_counts(zone_id)
            self.map_view.viewport().update()
    
    def add_zone(self):
        """Add a new zone with default letter name."""
        # Generate next zone letter (A, B, C, ...)
        zone_letter = chr(ord('A') + self.zone_counter)
        self.zone_counter += 1
        
        # Generate unique zone ID
        zone_id = f"zone_{self.zone_counter}"
        
        # Generate unique color for zone
        colors = [
            QColor(255, 100, 100),  # Red
            QColor(100, 255, 100),  # Green
            QColor(100, 100, 255),  # Blue
            QColor(255, 255, 100),  # Yellow
            QColor(255, 100, 255),  # Magenta
            QColor(100, 255, 255),  # Cyan
            QColor(255, 165, 0),    # Orange
            QColor(128, 0, 128),    # Purple
        ]
        zone_color = colors[(self.zone_counter - 1) % len(colors)]
        
        # Create zone widget
        zone_widget = self.create_zone_widget(zone_id, zone_letter, zone_color)
        
        # Store zone data
        self.zones[zone_id] = {
            'name': zone_letter,
            'color': zone_color,
            'widget': zone_widget
        }
        
        # Add zone to map view
        self.map_view.zones[zone_id] = {
            'areas': [],
            'name': zone_letter,
            'color': zone_color
        }
        
        # Add widget to container
        self.zones_container.addWidget(zone_widget)
        
        # Set this zone as active for selection
        self.current_zone_id = zone_id
        self.map_view.set_selection_mode('zone', zone_id)
        
        # Initialize counts
        self.update_zone_counts(zone_id)
    
    def create_zone_widget(self, zone_id: str, default_name: str, zone_color: QColor) -> QWidget:
        """Create a zone widget with editable name."""
        zone_frame = QFrame()
        zone_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 5px;
            }
        """)
        zone_layout = QHBoxLayout()
        zone_layout.setSpacing(10)
        zone_layout.setContentsMargins(5, 5, 5, 5)
        
        # Color indicator
        color_label = QLabel()
        color_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgb({zone_color.red()}, {zone_color.green()}, {zone_color.blue()});
                border: 1px solid #333;
                border-radius: 3px;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }}
        """)
        zone_layout.addWidget(color_label)
        
        # Zone name input
        name_input = QLineEdit(default_name)
        name_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                font-weight: bold;
            }
        """)
        name_input.textChanged.connect(lambda text: self.update_zone_name(zone_id, text))
        name_input.setObjectName(f"name_input_{zone_id}")  # For easy access
        zone_layout.addWidget(name_input)
        
        # Count label (roads and junctions)
        count_label = QLabel("0 roads, 0 junctions")
        count_label.setStyleSheet("color: #666; font-size: 11px;")
        count_label.setObjectName(f"count_{zone_id}")  # For easy access
        zone_layout.addWidget(count_label)
        
        zone_layout.addStretch()
        
        # Display button (toggle highlighting)
        display_btn = QPushButton("Display")
        display_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        display_btn.setCheckable(True)  # Make it toggleable
        display_btn.setObjectName(f"display_btn_{zone_id}")  # For easy access
        display_btn.clicked.connect(lambda checked: self.toggle_zone_display(zone_id, checked))
        zone_layout.addWidget(display_btn)
        
        # Delete button
        delete_btn = QPushButton("×")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 16px;
                min-width: 25px;
                max-width: 25px;
                min-height: 25px;
                max-height: 25px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        delete_btn.clicked.connect(lambda: self.delete_zone(zone_id))
        zone_layout.addWidget(delete_btn)
        
        zone_frame.setLayout(zone_layout)
        return zone_frame
    
    def update_zone_name(self, zone_id: str, new_name: str):
        """Update zone name in both widget and map view."""
        if zone_id in self.zones:
            self.zones[zone_id]['name'] = new_name
            if zone_id in self.map_view.zones:
                self.map_view.zones[zone_id]['name'] = new_name
            self.map_view.viewport().update()
    
    def update_zone_counts(self, zone_id: str):
        """Update the road and junction counts for a zone."""
        if zone_id not in self.zones or not self.network_parser:
            return
        
        # Get all areas for this zone
        zone_areas = self.map_view.zones.get(zone_id, {}).get('areas', [])
        if not zone_areas:
            # Update count label to show 0
            self._update_count_label(zone_id, 0, 0)
            return
        
        # Count edges and nodes within zone areas
        # Note: Roads (edges) and junctions can only belong to a single zone (first assignment wins).
        edges_in_zone = set()
        nodes_in_zone = set()
        
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        
        # Get all edges and junctions already assigned to other zones
        assigned_edges, assigned_junctions = self._get_assigned_edges_and_junctions_for_other_zones(zone_id)
        
        for zone_rect in zone_areas:
            # First, collect all nodes and junctions within this zone area
            # Note: edges reference junctions (or nodes) by ID, so we check both
            nodes_in_this_area = set()
            junctions_in_this_area = set()
            
            # Check nodes
            for node_id, node_data in nodes.items():
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                # Check if node is within the zone rectangle
                if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                    zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                    nodes_in_this_area.add(node_id)
                    nodes_in_zone.add(node_id)
            
            # Check junctions - only include if not already assigned to another zone
            for junction_id, junction_data in junctions.items():
                # Skip if junction already assigned to another zone
                if junction_id in assigned_junctions:
                    continue
                
                x = junction_data.get('x', 0)
                y = junction_data.get('y', 0)
                # Check if junction is within the zone rectangle
                if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                    zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                    junctions_in_this_area.add(junction_id)
                    nodes_in_zone.add(junction_id)  # Junctions are also counted as nodes
            
            # Check each edge - include only if BOTH endpoints are in the zone
            # AND the edge hasn't been assigned to another zone yet
            for edge_id, edge_data in edges.items():
                # Skip if edge already assigned to another zone
                if edge_id in assigned_edges:
                    continue
                
                from_node = edge_data.get('from')
                to_node = edge_data.get('to')
                
                # Both endpoints must be in the zone area
                if from_node and to_node:
                    # Check if both endpoints are in either nodes or junctions
                    from_in_zone = (from_node in nodes_in_this_area or from_node in junctions_in_this_area)
                    to_in_zone = (to_node in nodes_in_this_area or to_node in junctions_in_this_area)
                    
                    if from_in_zone and to_in_zone:
                        edges_in_zone.add(edge_id)
                        # Also add the nodes/junctions (already added above, but ensure they're in the set)
                        nodes_in_zone.add(from_node)
                        nodes_in_zone.add(to_node)
        
        # Update count label
        self._update_count_label(zone_id, len(edges_in_zone), len(nodes_in_zone))
        
        # Update visualization to highlight selected edges and nodes
        self._update_zone_visualization()
    
    def _update_count_label(self, zone_id: str, edge_count: int, node_count: int):
        """Update the count label in the zone widget."""
        if zone_id not in self.zones:
            return
        
        widget = self.zones[zone_id]['widget']
        count_label = widget.findChild(QLabel, f"count_{zone_id}")
        if count_label:
            count_label.setText(f"{edge_count} roads, {node_count} junctions")
    
    def _get_assigned_edges_and_junctions_for_other_zones(self, current_zone_id: str) -> tuple:
        """Get all edges and junctions that have been assigned to other zones.
        
        This ensures that roads and junctions can only belong to a single zone (first assignment wins).
        
        Returns:
            tuple: (assigned_edges, assigned_junctions) sets
        """
        if not self.network_parser:
            return set(), set()
        
        assigned_edges = set()
        assigned_junctions = set()
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        
        # Process all other zones first (before current zone)
        for zone_id, zone_data in self.map_view.zones.items():
            if zone_id == current_zone_id:
                continue  # Skip current zone
            
            zone_areas = zone_data.get('areas', [])
            if not zone_areas:
                continue
            
            for zone_rect in zone_areas:
                # Collect nodes and junctions in this area
                nodes_in_this_area = set()
                junctions_in_this_area = set()
                
                # Check nodes
                for node_id, node_data in nodes.items():
                    x = node_data.get('x', 0)
                    y = node_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        nodes_in_this_area.add(node_id)
                
                # Check junctions - mark as assigned if in this zone
                for junction_id, junction_data in junctions.items():
                    if junction_id in assigned_junctions:
                        continue  # Already assigned
                    
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        junctions_in_this_area.add(junction_id)
                        assigned_junctions.add(junction_id)  # Mark as assigned
                
                # Check edges - mark as assigned if both endpoints are in this zone
                for edge_id, edge_data in edges.items():
                    if edge_id in assigned_edges:
                        continue  # Already assigned
                    
                    from_node = edge_data.get('from')
                    to_node = edge_data.get('to')
                    
                    if from_node and to_node:
                        from_in_zone = (from_node in nodes_in_this_area or from_node in junctions_in_this_area)
                        to_in_zone = (to_node in nodes_in_this_area or to_node in junctions_in_this_area)
                        
                        if from_in_zone and to_in_zone:
                            assigned_edges.add(edge_id)
        
        return assigned_edges, assigned_junctions
    
    def _get_all_assigned_edges_and_junctions(self) -> tuple:
        """Get all edges and junctions that have been assigned to any zone.
        
        Returns:
            tuple: (assigned_edges, assigned_junctions) sets
        """
        if not self.network_parser:
            return set(), set()
        
        assigned_edges = set()
        assigned_junctions = set()
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        
        # Process all zones
        for zone_id, zone_data in self.map_view.zones.items():
            zone_areas = zone_data.get('areas', [])
            if not zone_areas:
                continue
            
            for zone_rect in zone_areas:
                # Collect nodes and junctions in this area
                nodes_in_this_area = set()
                junctions_in_this_area = set()
                
                # Check nodes
                for node_id, node_data in nodes.items():
                    x = node_data.get('x', 0)
                    y = node_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        nodes_in_this_area.add(node_id)
                
                # Check junctions - mark as assigned if in this zone
                for junction_id, junction_data in junctions.items():
                    if junction_id in assigned_junctions:
                        continue  # Already assigned
                    
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        junctions_in_this_area.add(junction_id)
                        assigned_junctions.add(junction_id)  # Mark as assigned
                
                # Check edges - mark as assigned if both endpoints are in this zone
                for edge_id, edge_data in edges.items():
                    if edge_id in assigned_edges:
                        continue  # Already assigned
                    
                    from_node = edge_data.get('from')
                    to_node = edge_data.get('to')
                    
                    if from_node and to_node:
                        from_in_zone = (from_node in nodes_in_this_area or from_node in junctions_in_this_area)
                        to_in_zone = (to_node in nodes_in_this_area or to_node in junctions_in_this_area)
                        
                        if from_in_zone and to_in_zone:
                            assigned_edges.add(edge_id)
        
        return assigned_edges, assigned_junctions
    
    def _create_zone_for_unselected_items(self, unselected_edges: set, unselected_junctions: set):
        """Create a new zone for unselected edges and junctions."""
        if not self.network_parser:
            return
        
        # Generate next zone letter
        zone_letter = chr(ord('A') + self.zone_counter)
        self.zone_counter += 1
        
        # Generate unique zone ID
        zone_id = f"zone_{self.zone_counter}"
        
        # Generate unique color for zone
        colors = [
            QColor(255, 100, 100),  # Red
            QColor(100, 255, 100),  # Green
            QColor(100, 100, 255),  # Blue
            QColor(255, 255, 100),  # Yellow
            QColor(255, 100, 255),  # Magenta
            QColor(100, 255, 255),  # Cyan
            QColor(255, 165, 0),    # Orange
            QColor(128, 0, 128),    # Purple
        ]
        zone_color = colors[(self.zone_counter - 1) % len(colors)]
        
        # Get junctions to create area rectangles
        junctions = self.network_parser.get_junctions()
        edges = self.network_parser.get_edges()
        
        # Collect all junction positions for unselected junctions
        junction_positions = []
        for junction_id in unselected_junctions:
            if junction_id in junctions:
                junction_data = junctions[junction_id]
                x = junction_data.get('x', 0)
                y = junction_data.get('y', 0)
                junction_positions.append((x, y))
        
        # Also collect junction positions from unselected edges
        for edge_id in unselected_edges:
            if edge_id in edges:
                edge_data = edges[edge_id]
                from_node = edge_data.get('from')
                to_node = edge_data.get('to')
                
                # Add from junction if it's not already assigned
                if from_node and from_node in junctions:
                    junction_data = junctions[from_node]
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    if (x, y) not in junction_positions:
                        junction_positions.append((x, y))
                
                # Add to junction if it's not already assigned
                if to_node and to_node in junctions:
                    junction_data = junctions[to_node]
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    if (x, y) not in junction_positions:
                        junction_positions.append((x, y))
        
        # Create area rectangles that cover all unselected junctions
        zone_areas = []
        if junction_positions:
            # Calculate bounding box for all unselected junctions
            xs = [pos[0] for pos in junction_positions]
            ys = [pos[1] for pos in junction_positions]
            
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Add some padding
                padding = 50.0
                area_rect = QRectF(
                    min_x - padding,
                    min_y - padding,
                    (max_x - min_x) + 2 * padding,
                    (max_y - min_y) + 2 * padding
                )
                zone_areas.append(area_rect)
        
        # Create zone widget
        zone_widget = self.create_zone_widget(zone_id, zone_letter, zone_color)
        
        # Store zone data
        self.zones[zone_id] = {
            'name': zone_letter,
            'color': zone_color,
            'widget': zone_widget
        }
        
        # Add zone to map view
        self.map_view.zones[zone_id] = {
            'areas': zone_areas,
            'name': zone_letter,
            'color': zone_color
        }
        
        # Add widget to container
        self.zones_container.addWidget(zone_widget)
        
        # Update counts for the new zone
        self.update_zone_counts(zone_id)
        
        # Update visualization
        self._update_zone_visualization()
        
        # Update map view
        self.map_view.viewport().update()
    
    def _update_zone_visualization(self):
        """Update visualization to highlight all selected edges and nodes across all zones.
        
        Note: Roads (edges) and junctions can only belong to a single zone (first assignment wins).
        """
        if not self.network_parser:
            return
        
        # Collect all selected edges and nodes from all zones
        all_selected_edges = set()
        all_selected_nodes = set()
        assigned_edges = set()  # Track edges that have already been assigned to a zone
        assigned_junctions = set()  # Track junctions that have already been assigned to a zone
        
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        
        # Process zones in order (first zone gets priority for edges and junctions)
        # Get all zone areas - process in order
        for zone_id, zone_data in self.map_view.zones.items():
            zone_areas = zone_data.get('areas', [])
            if not zone_areas:
                continue
            
            # Count edges and nodes within zone areas
            for zone_rect in zone_areas:
                # First, collect all nodes and junctions within this zone area
                nodes_in_this_area = set()
                junctions_in_this_area = set()
                
                # Check nodes
                for node_id, node_data in nodes.items():
                    x = node_data.get('x', 0)
                    y = node_data.get('y', 0)
                    # Check if node is within the zone rectangle
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        nodes_in_this_area.add(node_id)
                        all_selected_nodes.add(node_id)  # Nodes can be in multiple zones
                
                # Check junctions - only include if not already assigned to another zone
                for junction_id, junction_data in junctions.items():
                    # Skip if junction already assigned to another zone
                    if junction_id in assigned_junctions:
                        continue
                    
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    # Check if junction is within the zone rectangle
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        junctions_in_this_area.add(junction_id)
                        all_selected_nodes.add(junction_id)
                        assigned_junctions.add(junction_id)  # Mark as assigned
                
                # Check each edge - include only if BOTH endpoints are in the zone
                # AND the edge hasn't been assigned to another zone yet
                for edge_id, edge_data in edges.items():
                    # Skip if edge already assigned to another zone
                    if edge_id in assigned_edges:
                        continue
                    
                    from_node = edge_data.get('from')
                    to_node = edge_data.get('to')
                    
                    # Both endpoints must be in the zone area
                    if from_node and to_node:
                        # Check if both endpoints are in either nodes or junctions
                        from_in_zone = (from_node in nodes_in_this_area or from_node in junctions_in_this_area)
                        to_in_zone = (to_node in nodes_in_this_area or to_node in junctions_in_this_area)
                        
                        if from_in_zone and to_in_zone:
                            # Assign edge to this zone (first assignment wins)
                            all_selected_edges.add(edge_id)
                            assigned_edges.add(edge_id)
                            # Also add the nodes/junctions
                            all_selected_nodes.add(from_node)
                            all_selected_nodes.add(to_node)
        
        # Update visualization
        self.map_view.set_selected_edges(all_selected_edges)
        self.map_view.set_selected_nodes(all_selected_nodes)
    
    def toggle_zone_display(self, zone_id: str, checked: bool):
        """Toggle display (highlighting) of roads and junctions for a zone."""
        if checked:
            # Add to displayed zones
            self.displayed_zones.add(zone_id)
        else:
            # Remove from displayed zones
            self.displayed_zones.discard(zone_id)
        
        # Update button state to reflect checked status
        if zone_id in self.zones:
            widget = self.zones[zone_id]['widget']
            display_btn = widget.findChild(QPushButton, f"display_btn_{zone_id}")
            if display_btn:
                display_btn.setChecked(checked)
        
        # Update visualization to highlight displayed zones
        self._update_display_visualization()
    
    def _update_display_visualization(self):
        """Update visualization to highlight roads and junctions for displayed zones."""
        if not self.network_parser:
            return
        
        # Collect all edges and junctions from displayed zones
        displayed_edges = set()
        displayed_nodes = set()
        
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        
        # Process all zones in order to determine assignments (first assignment wins)
        # Then collect items that belong to displayed zones
        assigned_edges_to_zone = {}  # edge_id -> zone_id
        assigned_junctions_to_zone = {}  # junction_id -> zone_id
        
        # First pass: assign edges and junctions to zones (respecting first assignment rule)
        for zone_id, zone_data in self.map_view.zones.items():
            zone_areas = zone_data.get('areas', [])
            if not zone_areas:
                continue
            
            # First, collect all nodes and junctions in all areas of this zone
            nodes_in_zone = set()
            junctions_in_zone = set()
            
            for zone_rect in zone_areas:
                # Check nodes
                for node_id, node_data in nodes.items():
                    x = node_data.get('x', 0)
                    y = node_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        nodes_in_zone.add(node_id)
                
                # Check junctions - assign to this zone if not already assigned
                for junction_id, junction_data in junctions.items():
                    if junction_id in assigned_junctions_to_zone:
                        continue  # Already assigned to another zone
                    
                    x = junction_data.get('x', 0)
                    y = junction_data.get('y', 0)
                    if (zone_rect.x() <= x <= zone_rect.x() + zone_rect.width() and
                        zone_rect.y() <= y <= zone_rect.y() + zone_rect.height()):
                        assigned_junctions_to_zone[junction_id] = zone_id
                        junctions_in_zone.add(junction_id)
            
            # Now check edges - assign to this zone if both endpoints are in the zone
            for edge_id, edge_data in edges.items():
                if edge_id in assigned_edges_to_zone:
                    continue  # Already assigned to another zone
                
                from_node = edge_data.get('from')
                to_node = edge_data.get('to')
                
                if from_node and to_node:
                    # Check if both endpoints are in this zone
                    # from_node can be in nodes_in_zone OR assigned to this zone's junctions
                    from_in_zone = (from_node in nodes_in_zone or 
                                   (from_node in assigned_junctions_to_zone and 
                                    assigned_junctions_to_zone[from_node] == zone_id))
                    to_in_zone = (to_node in nodes_in_zone or 
                                 (to_node in assigned_junctions_to_zone and 
                                  assigned_junctions_to_zone[to_node] == zone_id))
                    
                    if from_in_zone and to_in_zone:
                        assigned_edges_to_zone[edge_id] = zone_id
        
        # Second pass: collect items that belong to displayed zones
        for zone_id in self.displayed_zones:
            # Collect edges assigned to this zone
            for edge_id, assigned_zone_id in assigned_edges_to_zone.items():
                if assigned_zone_id == zone_id:
                    displayed_edges.add(edge_id)
                    # Also add the endpoints
                    if edge_id in edges:
                        edge_data = edges[edge_id]
                        if edge_data.get('from'):
                            displayed_nodes.add(edge_data['from'])
                        if edge_data.get('to'):
                            displayed_nodes.add(edge_data['to'])
            
            # Collect junctions assigned to this zone
            for junction_id, assigned_zone_id in assigned_junctions_to_zone.items():
                if assigned_zone_id == zone_id:
                    displayed_nodes.add(junction_id)
        
        # Update visualization
        self.map_view.set_selected_edges(displayed_edges)
        self.map_view.set_selected_nodes(displayed_nodes)
    
    def delete_zone(self, zone_id: str):
        """Delete a zone."""
        if self.zones_locked:
            QMessageBox.warning(self, "Zones Locked", "Cannot delete zones after clicking Done.")
            return
        
        if zone_id in self.zones:
            reply = QMessageBox.question(
                self, "Delete Zone",
                f"Are you sure you want to delete zone '{self.zones[zone_id]['name']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove widget
                widget = self.zones[zone_id]['widget']
                self.zones_container.removeWidget(widget)
                widget.deleteLater()
                
                # Remove from zones dict
                del self.zones[zone_id]
                
                # Remove from map view
                if zone_id in self.map_view.zones:
                    del self.map_view.zones[zone_id]
                
                # Remove from displayed zones
                self.displayed_zones.discard(zone_id)
                
                # Reset selection if this was the active zone
                if self.current_zone_id == zone_id:
                    self.current_zone_id = None
                    self.map_view.set_selection_mode(None)
                
                # Update display visualization if needed
                self._update_display_visualization()
                
                self.map_view.viewport().update()
    
    def lock_zones(self):
        """Lock zones: disable editing, remove colored areas, reset highlighting."""
        if self.zones_locked:
            return
        
        if not self.network_parser:
            QMessageBox.warning(self, "No Network", "Cannot lock zones: no network loaded.")
            return
        
        # Check for unselected roads and junctions
        all_edges = set(self.network_parser.get_edges().keys())
        all_junctions = set(self.network_parser.get_junctions().keys())
        
        # Get all assigned edges and junctions from existing zones
        assigned_edges, assigned_junctions = self._get_all_assigned_edges_and_junctions()
        
        unselected_edges = all_edges - assigned_edges
        unselected_junctions = all_junctions - assigned_junctions
        
        # If there are unselected items, prompt user
        if unselected_edges or unselected_junctions:
            edge_count = len(unselected_edges)
            junction_count = len(unselected_junctions)
            
            reply = QMessageBox.question(
                self, "Unselected Items",
                f"There are {edge_count} unselected roads and {junction_count} unselected junctions.\n\n"
                "Would you like to add them all to a new zone?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Create a new zone for unselected items
                self._create_zone_for_unselected_items(unselected_edges, unselected_junctions)
        
        # Lock zone names (disable editing)
        for zone_id, zone_data in self.zones.items():
            widget = zone_data['widget']
            name_input = widget.findChild(QLineEdit, f"name_input_{zone_id}")
            if name_input:
                name_input.setEnabled(False)
        
        # Disable add zone button
        add_zone_btn = self.findChild(QPushButton)
        if add_zone_btn:
            # Find the + button in the zones header
            for child in self.findChildren(QPushButton):
                if child.text() == "+":
                    child.setEnabled(False)
                    break
        
        # Disable delete buttons (but keep Display buttons enabled)
        for zone_id, zone_data in self.zones.items():
            widget = zone_data['widget']
            for child in widget.findChildren(QPushButton):
                if child.text() == "×":
                    child.setEnabled(False)
                # Display button remains enabled even after locking
        
        # Disable done button
        if hasattr(self, 'done_btn'):
            self.done_btn.setEnabled(False)
        
        # Reset selection mode
        self.current_zone_id = None
        self.map_view.set_selection_mode(None)
        
        # Reset edge and node highlighting (reverse thickening)
        self.map_view.set_selected_edges(set())
        self.map_view.set_selected_nodes(set())
        
        # Mark zones as locked
        self.zones_locked = True
        
        # Update map view to show locked state (zone names in centers)
        self.map_view.viewport().update()
        
        QMessageBox.information(self, "Zones Locked", "Zone configuration is now locked.")
