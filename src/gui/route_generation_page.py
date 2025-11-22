"""
Route Generation page for creating SUMO route files manually.
"""

import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QGuiApplication, QPainter,
                           QPen)
from PySide6.QtWidgets import (QFrame, QGraphicsView, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

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
            
            # Check if this zone is being displayed (darkened)
            is_displayed = False
            if hasattr(self, 'route_page'):
                is_displayed = zone_id in self.route_page.displayed_zones
            
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
                    # Draw zone area - darker if being displayed
                    if is_displayed:
                        # Darker fill for displayed zones
                        painter.setPen(QPen(zone_color, 3))
                        painter.setBrush(QBrush(QColor(zone_color.red(), zone_color.green(), zone_color.blue(), 80)))
                    else:
                        # Normal fill
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
        self.zones = {}  # Dict of {zone_id: {'name': str, 'color': QColor, 'widget': QWidget, 'areas': [QRectF]}}
        self.zone_counter = 0  # Counter for zone letters (A, B, C, ...)
        self.current_zone_id = None  # Currently active zone for selection
        self.zones_locked = False  # Whether zones are locked (done button clicked)
        self.displayed_zones = set()  # Set of zone IDs that are currently being displayed (highlighted)
        
        # Track assignments: junction_id -> zone_id, edge_id -> zone_id
        self.junction_assignments = {}  # Dict of {junction_id: zone_id}
        self.road_assignments = {}  # Dict of {edge_id: zone_id}
        
        # Check if Porto mode is enabled
        self.porto_mode_enabled = self.config_manager.get_use_porto_dataset()
        
        # Porto-specific paths (only if Porto mode enabled)
        if self.porto_mode_enabled:
            porto_dataset_path = self.config_manager.get_porto_dataset_path()
            if porto_dataset_path:
                self.porto_dataset_path = Path(porto_dataset_path)
            else:
                # Fallback to default path
                project_path_obj = Path(project_path).resolve()
                workspace_root = project_path_obj
                while workspace_root != workspace_root.parent:
                    if (workspace_root / 'Porto').exists():
                        break
                    workspace_root = workspace_root.parent
                if not (workspace_root / 'Porto').exists():
                    workspace_root = Path('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester')
                self.porto_dataset_path = workspace_root / 'Porto' / 'dataset' / 'train.csv'
        
        self.init_ui()
        self.load_network()
        # Load saved zones after network is loaded (but before Porto neighborhoods)
        # This ensures saved zones are loaded first, then Porto neighborhoods are added if needed
        if hasattr(self, 'zones_container'):
            self.load_saved_zones()
    
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
        content_layout.setContentsMargins(0, 0, 0, 0)  # Prevent overflow
        
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
        
        # Right side: Route Configuration (scrollable)
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        config_scroll.setSizePolicy(QWidget.Expanding, QWidget.Expanding)
        config_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        
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
        # Prevent horizontal overflow - ensure it respects available width
        config_group.setSizePolicy(QWidget.Expanding, QWidget.Preferred)
        config_layout = QVBoxLayout()
        config_layout.setSpacing(15)
        config_layout.setContentsMargins(10, 10, 10, 10)
        
        # Porto Dataset Conversion Section (only if Porto mode enabled)
        if self.porto_mode_enabled:
            self.add_porto_conversion_section(config_layout)
        
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
        
        # Statistics label showing total and selected roads/junctions
        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setStyleSheet("color: #666; padding: 5px;")
        self.stats_label.setWordWrap(True)
        self.stats_label.setSizePolicy(QWidget.Expanding, QWidget.Preferred)
        zones_header.addWidget(self.stats_label)
        
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
        self.done_btn.clicked.connect(self.on_done_unlock_clicked)
        zones_header.addWidget(self.done_btn)
        
        config_layout.addLayout(zones_header)
        
        # Zones list/container (scrollable with max height = 50% of screen)
        zones_scroll = QScrollArea()
        zones_scroll.setWidgetResizable(True)
        zones_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        zones_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        zones_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: #fafafa;
            }
        """)
        
        # Get screen height and set max height to 50%
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_height = screen.geometry().height()
            max_zones_height = int(screen_height * 0.5)
            zones_scroll.setMaximumHeight(max_zones_height)
        
        zones_widget = QWidget()
        self.zones_container = QVBoxLayout()
        self.zones_container.setSpacing(8)  # Reduced spacing
        self.zones_container.setContentsMargins(5, 5, 5, 5)
        zones_widget.setLayout(self.zones_container)
        zones_scroll.setWidget(zones_widget)
        
        config_layout.addWidget(zones_scroll)
        
        config_layout.addStretch()
        config_group.setLayout(config_layout)
        config_scroll.setWidget(config_group)
        
        # Set maximum width to prevent overflow - use stretch but respect available space
        config_scroll.setSizePolicy(QWidget.Expanding, QWidget.Expanding)
        config_scroll.setMaximumWidth(500)  # Limit max width to prevent overflow
        content_layout.addWidget(config_scroll, stretch=1)
        
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
                        # Update statistics when network is loaded
                        self.update_statistics()
                        
                        # Load Porto neighborhoods if Porto mode is enabled
                        if self.porto_mode_enabled:
                            self.load_porto_neighborhoods()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load network: {str(e)}")
    
    def on_area_selected(self, area_type: str, rect: QRectF):
        """Handle area selection."""
        # Placeholder for future implementation
        pass
    
    def on_zone_area_selected(self, zone_id: str, rect: QRectF):
        """Handle zone area selection and assign junctions and roads."""
        if zone_id not in self.zones or not self.network_parser:
            return
        
        # Update zone areas in zones dict
        if 'areas' not in self.zones[zone_id]:
            self.zones[zone_id]['areas'] = []
        if rect not in self.zones[zone_id]['areas']:
            self.zones[zone_id]['areas'].append(rect)
        
        # Ensure map_view.zones is also updated
        if zone_id not in self.map_view.zones:
            self.map_view.zones[zone_id] = {
                'areas': [],
                'name': self.zones[zone_id].get('name', zone_id),
                'color': self.zones[zone_id].get('color', QColor(200, 200, 200))
            }
        if 'areas' not in self.map_view.zones[zone_id]:
            self.map_view.zones[zone_id]['areas'] = []
        if rect not in self.map_view.zones[zone_id]['areas']:
            self.map_view.zones[zone_id]['areas'].append(rect)
        
        # Assign junctions and roads based on the new rectangle
        self._assign_junctions_and_roads_for_area(zone_id, rect)
        
        # Update counts in widget and visualization
        self.update_zone_counts(zone_id)
        self.map_view.viewport().update()
        # Update statistics
        self.update_statistics()
        
        # Save zones after adding area
        self.save_zones()
    
    def update_statistics(self):
        """Update the statistics label showing total and selected roads/junctions."""
        if not self.network_parser:
            self.stats_label.setText("")
            return
        
        # Get total counts
        all_edges = self.network_parser.get_edges()
        all_junctions = self.network_parser.get_junctions()
        total_roads = len(all_edges)
        total_junctions = len(all_junctions)
        
        # Get selected counts (assigned to zones)
        selected_roads = len(self.road_assignments)
        selected_junctions = len(self.junction_assignments)
        
        # Update label
        stats_text = f"Total: {total_roads} roads, {total_junctions} junctions | Assigned: {selected_roads} roads, {selected_junctions} junctions"
        self.stats_label.setText(stats_text)
    
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
            'widget': zone_widget,
            'areas': []
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
        
        # Update statistics
        self.update_statistics()
        
        # Save zones after adding
        self.save_zones()
    
    def create_zone_widget(self, zone_id: str, default_name: str, zone_color: QColor) -> QWidget:
        """Create a zone widget with editable name."""
        zone_frame = QFrame()
        zone_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 3px;
            }
        """)
        # Reduce height by 30% - reduce padding and margins
        zone_frame.setMaximumHeight(35)  # Reduced from ~50px to ~35px (30% reduction)
        zone_layout = QHBoxLayout()
        zone_layout.setSpacing(8)  # Reduced spacing
        zone_layout.setContentsMargins(4, 3, 4, 3)  # Reduced margins
        
        # Color indicator (smaller)
        color_label = QLabel()
        color_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgb({zone_color.red()}, {zone_color.green()}, {zone_color.blue()});
                border: 1px solid #333;
                border-radius: 2px;
                min-width: 16px;
                max-width: 16px;
                min-height: 16px;
                max-height: 16px;
            }}
        """)
        zone_layout.addWidget(color_label)
        
        # Zone name input (smaller)
        name_input = QLineEdit(default_name)
        name_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 2px;
                padding: 2px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        name_input.setMaximumHeight(22)  # Reduced height
        name_input.textChanged.connect(lambda text: self.update_zone_name(zone_id, text))
        name_input.setObjectName(f"name_input_{zone_id}")  # For easy access
        zone_layout.addWidget(name_input)
        
        # Count label (roads and junctions) - smaller font
        count_label = QLabel("0 roads, 0 junctions")
        count_label.setStyleSheet("color: #666; font-size: 9px;")
        count_label.setObjectName(f"count_{zone_id}")  # For easy access
        zone_layout.addWidget(count_label)
        
        zone_layout.addStretch()
        
        # Display button (toggle highlighting) - smaller
        display_btn = QPushButton("Display")
        display_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 2px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        display_btn.setMaximumHeight(22)  # Reduced height
        display_btn.setCheckable(True)  # Make it toggleable
        display_btn.setObjectName(f"display_btn_{zone_id}")  # For easy access
        display_btn.clicked.connect(lambda checked: self.toggle_zone_display(zone_id, checked))
        zone_layout.addWidget(display_btn)
        
        # Delete button (smaller)
        delete_btn = QPushButton("×")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 2px;
                font-weight: bold;
                font-size: 14px;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
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
            # Save zones after name change
            self.save_zones()
    
    def _assign_junctions_and_roads_for_area(self, zone_id: str, rect: QRectF):
        """Assign junctions and roads to a zone based on a newly selected area rectangle.
        
        Rules:
        1. All newly selected junctions within the selected area are assigned to the zone.
           CONSTRAINT: Only assign junctions that are located within the selected area boundaries.
        2. All newly selected roads which both start and end junctions are within the selected area are assigned to the zone.
           CONSTRAINT: Only assign roads that both junctions are located within the selected area boundaries.
        3. A selected road that only one of its junctions is already assigned to a different zone OR both its junctions are already assigned to the current zone is assigned to the zone.
           CONSTRAINT: Both junctions must still be within the selected area boundaries.
        """
        if not self.network_parser:
            return
        
        junctions = self.network_parser.get_junctions()
        edges = self.network_parser.get_edges()
        nodes = self.network_parser.get_nodes()
        
        # Rule 1: Assign all newly selected junctions within the rectangle
        # CONSTRAINT: Only assign junctions that are located within the selected area boundaries
        for junction_id, junction_data in junctions.items():
            # Skip if already assigned to another zone
            if junction_id in self.junction_assignments and self.junction_assignments[junction_id] != zone_id:
                continue
            
            x = junction_data.get('x', 0)
            y = junction_data.get('y', 0)
            # CONSTRAINT: Only assign if junction is within rectangle boundaries
            if rect.contains(QPointF(x, y)):
                # Assign junction to this zone
                self.junction_assignments[junction_id] = zone_id
        
        # Rule 2 & 3: Assign roads
        # CONSTRAINT: Only assign roads that both junctions are located within the selected area boundaries
        for edge_id, edge_data in edges.items():
            # Skip if already assigned to any zone (cannot reassign to a different zone)
            if edge_id in self.road_assignments:
                if self.road_assignments[edge_id] == zone_id:
                    continue  # Already assigned to this zone, skip
                else:
                    continue  # Already assigned to a different zone, cannot reassign
            
            from_node_id = edge_data.get('from')
            to_node_id = edge_data.get('to')
            
            if not from_node_id or not to_node_id:
                continue
            
            # Get junction/node positions and assignment status
            from_in_rect = False
            to_in_rect = False
            from_assigned_to_other = False
            to_assigned_to_other = False
            from_assigned_to_current = False
            to_assigned_to_current = False
            
            # Check from_node
            if from_node_id in junctions:
                junction_data = junctions[from_node_id]
                x = junction_data.get('x', 0)
                y = junction_data.get('y', 0)
                from_in_rect = rect.contains(QPointF(x, y))
                if from_node_id in self.junction_assignments:
                    assigned_zone = self.junction_assignments[from_node_id]
                    from_assigned_to_other = (assigned_zone != zone_id)
                    from_assigned_to_current = (assigned_zone == zone_id)
            elif from_node_id in nodes:
                node_data = nodes[from_node_id]
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                from_in_rect = rect.contains(QPointF(x, y))
            
            # Check to_node
            if to_node_id in junctions:
                junction_data = junctions[to_node_id]
                x = junction_data.get('x', 0)
                y = junction_data.get('y', 0)
                to_in_rect = rect.contains(QPointF(x, y))
                if to_node_id in self.junction_assignments:
                    assigned_zone = self.junction_assignments[to_node_id]
                    to_assigned_to_other = (assigned_zone != zone_id)
                    to_assigned_to_current = (assigned_zone == zone_id)
            elif to_node_id in nodes:
                node_data = nodes[to_node_id]
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                to_in_rect = rect.contains(QPointF(x, y))
            
            # CONSTRAINT: Both junctions must be within the selected area boundaries for any assignment
            if not (from_in_rect and to_in_rect):
                continue  # Skip this road if both junctions are not in rectangle
            
            # Both junctions are in rectangle - now check assignment rules
            
            # Rule 2: Both junctions are newly selected (not assigned to any zone)
            # OR both junctions are now assigned to current zone (just assigned in Rule 1)
            from_newly_selected = (from_node_id not in self.junction_assignments)
            to_newly_selected = (to_node_id not in self.junction_assignments)
            from_now_assigned_to_current = (from_node_id in self.junction_assignments and 
                                            self.junction_assignments[from_node_id] == zone_id)
            to_now_assigned_to_current = (to_node_id in self.junction_assignments and 
                                          self.junction_assignments[to_node_id] == zone_id)
            
            # Rule 2: Both junctions are newly selected OR both are now assigned to current zone
            if (from_newly_selected and to_newly_selected) or (from_now_assigned_to_current and to_now_assigned_to_current):
                self.road_assignments[edge_id] = zone_id
            # Rule 3: One junction is already assigned to a different zone OR both are assigned to current zone
            elif (from_assigned_to_other or to_assigned_to_other) or (from_assigned_to_current and to_assigned_to_current):
                self.road_assignments[edge_id] = zone_id
    
    def update_zone_counts(self, zone_id: str):
        """Update the road and junction counts for a zone based on assignments."""
        if zone_id not in self.zones:
            return
        
        # Count assigned junctions and roads for this zone
        assigned_junctions = [j_id for j_id, z_id in self.junction_assignments.items() if z_id == zone_id]
        assigned_roads = [e_id for e_id, z_id in self.road_assignments.items() if z_id == zone_id]
        
        # Update count label
        self._update_count_label(zone_id, len(assigned_roads), len(assigned_junctions))
        
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
        
        # Get junctions, nodes, and edges to create area rectangles
        junctions = self.network_parser.get_junctions()
        nodes = self.network_parser.get_nodes()
        edges = self.network_parser.get_edges()
        
        # Collect all positions (both nodes and junctions) that need to be included
        all_positions = []
        position_set = set()  # To avoid duplicates
        
        # Collect positions from unselected junctions
        for junction_id in unselected_junctions:
            if junction_id in junctions:
                junction_data = junctions[junction_id]
                x = junction_data.get('x', 0)
                y = junction_data.get('y', 0)
                pos = (x, y)
                if pos not in position_set:
                    position_set.add(pos)
                    all_positions.append(pos)
        
        # Collect positions from unselected edges (both from and to nodes/junctions)
        for edge_id in unselected_edges:
            if edge_id in edges:
                edge_data = edges[edge_id]
                from_node = edge_data.get('from')
                to_node = edge_data.get('to')
                
                # Check from_node - could be a junction or a regular node
                if from_node:
                    x, y = None, None
                    if from_node in junctions:
                        junction_data = junctions[from_node]
                        x = junction_data.get('x', 0)
                        y = junction_data.get('y', 0)
                    elif from_node in nodes:
                        node_data = nodes[from_node]
                        x = node_data.get('x', 0)
                        y = node_data.get('y', 0)
                    
                    if x is not None and y is not None:
                        pos = (x, y)
                        if pos not in position_set:
                            position_set.add(pos)
                            all_positions.append(pos)
                
                # Check to_node - could be a junction or a regular node
                if to_node:
                    x, y = None, None
                    if to_node in junctions:
                        junction_data = junctions[to_node]
                        x = junction_data.get('x', 0)
                        y = junction_data.get('y', 0)
                    elif to_node in nodes:
                        node_data = nodes[to_node]
                        x = node_data.get('x', 0)
                        y = node_data.get('y', 0)
                    
                    if x is not None and y is not None:
                        pos = (x, y)
                        if pos not in position_set:
                            position_set.add(pos)
                            all_positions.append(pos)
        
        # Create area rectangles that cover all unselected items
        zone_areas = []
        if all_positions:
            # Calculate bounding box for all positions
            xs = [pos[0] for pos in all_positions]
            ys = [pos[1] for pos in all_positions]
            
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Add padding to ensure all items are included
                # Use a percentage-based padding to handle large networks
                width = max_x - min_x
                height = max_y - min_y
                padding_x = max(50.0, width * 0.1)  # At least 50, or 10% of width
                padding_y = max(50.0, height * 0.1)  # At least 50, or 10% of height
                
                area_rect = QRectF(
                    min_x - padding_x,
                    min_y - padding_y,
                    width + 2 * padding_x,
                    height + 2 * padding_y
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
        
        # Assign unselected junctions and roads to this new zone
        # Since we're creating a zone specifically for unselected items, assign them all directly
        if zone_areas:
            # Use the first (and likely only) area rectangle
            bounding_rect = zone_areas[0]
            
            # Assign all unselected junctions to this zone
            for junction_id in unselected_junctions:
                if junction_id not in self.junction_assignments:  # Double-check it's still unselected
                    self.junction_assignments[junction_id] = zone_id
            
            # Assign all unselected roads to this zone (only if both junctions are unselected or in this zone)
            for edge_id in unselected_edges:
                if edge_id not in self.road_assignments:  # Double-check it's still unselected
                    if edge_id in edges:
                        edge_data = edges[edge_id]
                        from_node_id = edge_data.get('from')
                        to_node_id = edge_data.get('to')
                        
                        if from_node_id and to_node_id:
                            # Check if both junctions are in the bounding area
                            from_in_area = False
                            to_in_area = False
                            
                            if from_node_id in junctions:
                                junction_data = junctions[from_node_id]
                                x = junction_data.get('x', 0)
                                y = junction_data.get('y', 0)
                                from_in_area = bounding_rect.contains(QPointF(x, y))
                            elif from_node_id in nodes:
                                node_data = nodes[from_node_id]
                                x = node_data.get('x', 0)
                                y = node_data.get('y', 0)
                                from_in_area = bounding_rect.contains(QPointF(x, y))
                            
                            if to_node_id in junctions:
                                junction_data = junctions[to_node_id]
                                x = junction_data.get('x', 0)
                                y = junction_data.get('y', 0)
                                to_in_area = bounding_rect.contains(QPointF(x, y))
                            elif to_node_id in nodes:
                                node_data = nodes[to_node_id]
                                x = node_data.get('x', 0)
                                y = node_data.get('y', 0)
                                to_in_area = bounding_rect.contains(QPointF(x, y))
                            
                            # Assign road if both junctions are in the area
                            if from_in_area and to_in_area:
                                self.road_assignments[edge_id] = zone_id
        
        # Update counts for the new zone
        self.update_zone_counts(zone_id)
        
        # Update visualization
        self._update_zone_visualization()
        
        # Update map view
        self.map_view.viewport().update()
        
        # Update statistics
        self.update_statistics()
        
        # Save zones after creating new zone
        self.save_zones()
    
    def _update_zone_visualization(self):
        """Update visualization to highlight roads and junctions for zones."""
        if not self.network_parser:
            return
        
        # Collect all assigned edges and junctions from all zones
        all_assigned_edges = set(self.road_assignments.keys())
        all_assigned_junctions = set(self.junction_assignments.keys())
        
        # Update visualization
        self.map_view.set_selected_edges(all_assigned_edges)
        self.map_view.set_selected_nodes(all_assigned_junctions)
    
    def toggle_zone_display(self, zone_id: str, checked: bool):
        """Toggle display (highlighting) of roads and junctions for a zone."""
        if checked:
            # Add to displayed zones
            self.displayed_zones.add(zone_id)
        else:
            # Remove from displayed zones
            self.displayed_zones.discard(zone_id)
        
        # Update button text and state
        if zone_id in self.zones:
            widget = self.zones[zone_id]['widget']
            display_btn = widget.findChild(QPushButton, f"display_btn_{zone_id}")
            if display_btn:
                display_btn.setChecked(checked)
                # Update button text
                if checked:
                    display_btn.setText("Hide")
                else:
                    display_btn.setText("Display")
        
        # Update visualization to highlight displayed zones and darken zone areas
        self._update_display_visualization()
        self.map_view.viewport().update()  # Trigger repaint to show darker zones
    
    def _update_display_visualization(self):
        """Update visualization to highlight roads and junctions for displayed zones."""
        if not self.network_parser:
            return
        
        # Collect edges and junctions assigned to displayed zones
        displayed_edges = set()
        displayed_junctions = set()
        
        for zone_id in self.displayed_zones:
            # Get edges assigned to this zone
            for edge_id, assigned_zone_id in self.road_assignments.items():
                if assigned_zone_id == zone_id:
                    displayed_edges.add(edge_id)
            
            # Get junctions assigned to this zone
            for junction_id, assigned_zone_id in self.junction_assignments.items():
                if assigned_zone_id == zone_id:
                    displayed_junctions.add(junction_id)
        
        # Update visualization
        self.map_view.set_selected_edges(displayed_edges)
        self.map_view.set_selected_nodes(displayed_junctions)
    
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
                # Clear assignments for this zone
                # Remove all junctions assigned to this zone
                junctions_to_remove = [j_id for j_id, z_id in self.junction_assignments.items() if z_id == zone_id]
                for junction_id in junctions_to_remove:
                    del self.junction_assignments[junction_id]
                
                # Remove all roads assigned to this zone
                roads_to_remove = [e_id for e_id, z_id in self.road_assignments.items() if z_id == zone_id]
                for edge_id in roads_to_remove:
                    del self.road_assignments[edge_id]
                
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
                
                # Update visualization
                self._update_zone_visualization()
                self._update_display_visualization()
                
                self.map_view.viewport().update()
                
                # Update statistics
                self.update_statistics()
                
                # Save zones after deletion
                self.save_zones()
    
    def on_done_unlock_clicked(self):
        """Handle Done/Unlock button click."""
        if self.zones_locked:
            # Zones are locked, so unlock them
            self.unlock_zones()
        else:
            # Zones are not locked, so check for unselected items and lock them
            self.lock_zones()
    
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
        
        # Get all assigned edges and junctions
        assigned_edges = set(self.road_assignments.keys())
        assigned_junctions = set(self.junction_assignments.keys())
        
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
        
        # Change done button to "Unlock"
        if hasattr(self, 'done_btn'):
            self.done_btn.setText("Unlock")
            self.done_btn.setEnabled(True)  # Keep it enabled so user can unlock
        
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
        
        # Update statistics
        self.update_statistics()
        
        QMessageBox.information(self, "Zones Locked", "Zone configuration is now locked.")
    
    def unlock_zones(self):
        """Unlock zones: show warning and re-enable editing."""
        if not self.zones_locked:
            return
        
        # Show warning message
        reply = QMessageBox.warning(
            self, "Unlock Zones",
            "Changing the zones configuration might affect the simulation behavior.\n\n"
            "Are you sure you want to unlock the zones?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Re-enable zone name editing
        for zone_id, zone_data in self.zones.items():
            widget = zone_data['widget']
            name_input = widget.findChild(QLineEdit, f"name_input_{zone_id}")
            if name_input:
                name_input.setEnabled(True)
        
        # Re-enable add zone button
        for child in self.findChildren(QPushButton):
            if child.text() == "+":
                child.setEnabled(True)
                break
        
        # Re-enable delete buttons
        for zone_id, zone_data in self.zones.items():
            widget = zone_data['widget']
            for child in widget.findChildren(QPushButton):
                if child.text() == "×":
                    child.setEnabled(True)
        
        # Change unlock button back to "Done"
        if hasattr(self, 'done_btn'):
            self.done_btn.setText("Done")
        
        # Mark zones as unlocked
        self.zones_locked = False
        
        # Update map view to show unlocked state (zone names at corners)
        self.map_view.viewport().update()
    
    def load_porto_neighborhoods(self):
        """Load Porto quadrants (4x4 grid) as zones when Porto mode is enabled."""
        if not self.porto_mode_enabled or not self.network_parser:
            return
        
        from src.utils.porto_neighborhoods import get_porto_quadrants
        
        network_bounds = self.network_parser.bounds
        
        if not network_bounds:
            return
        
        # Generate 16 quadrants (4x4 grid)
        quadrants = get_porto_quadrants(network_bounds)
        
        for quadrant_name, rect, zone_color in quadrants:
            # Create zone ID
            zone_id = f"porto_quadrant_{quadrant_name}"
            
            # Check if zone already exists (from saved zones)
            if zone_id not in self.zones:
                # Create zone widget
                zone_widget = self.create_zone_widget(zone_id, quadrant_name, zone_color)
                
                # Store zone data
                self.zones[zone_id] = {
                    'name': quadrant_name,
                    'color': zone_color,
                    'widget': zone_widget,
                    'areas': [rect]
                }
                
                # Add zone to map view
                self.map_view.zones[zone_id] = {
                    'areas': [rect],
                    'name': quadrant_name,
                    'color': zone_color
                }
                
                # Add widget to container
                if hasattr(self, 'zones_container'):
                    self.zones_container.addWidget(zone_widget)
                
                # Update zone counter
                self.zone_counter += 1
        
        # Assign junctions and roads to quadrants based on their areas
        for quadrant_name, rect, _ in quadrants:
            zone_id = f"porto_quadrant_{quadrant_name}"
            if zone_id in self.zones:
                # Ensure the zone has areas in map_view
                if zone_id not in self.map_view.zones:
                    self.map_view.zones[zone_id] = {
                        'areas': [rect],
                        'name': quadrant_name,
                        'color': self.zones[zone_id].get('color')
                    }
                elif not self.map_view.zones[zone_id].get('areas'):
                    # If map_view zone exists but has no areas, update it
                    self.map_view.zones[zone_id]['areas'] = [rect]
                
                # Assign junctions and roads for this quadrant
                self._assign_junctions_and_roads_for_area(zone_id, rect)
                
                # Update counts
                self.update_zone_counts(zone_id)
        
        # Save zones after loading
        self.save_zones()
        self.update_statistics()
    
    
    def load_saved_zones(self):
        """Load saved zones from project configuration."""
        if not hasattr(self, 'zones_container'):
            return
        
        saved_zones = self.config_manager.load_zones()
        
        for zone_id, zone_data in saved_zones.items():
            # Skip if already loaded (e.g., Porto neighborhoods)
            if zone_id in self.zones:
                # Update areas if they exist
                if zone_data.get('areas'):
                    self.zones[zone_id]['areas'] = zone_data['areas']
                    self.map_view.zones[zone_id]['areas'] = zone_data['areas']
                continue
            
            # Create zone widget
            zone_widget = self.create_zone_widget(
                zone_id, 
                zone_data.get('name', 'Zone'),
                zone_data.get('color')
            )
            
            # Store zone data
            self.zones[zone_id] = {
                'name': zone_data.get('name', 'Zone'),
                'color': zone_data.get('color'),
                'widget': zone_widget,
                'areas': zone_data.get('areas', [])
            }
            
            # Add zone to map view
            self.map_view.zones[zone_id] = {
                'areas': zone_data.get('areas', []),
                'name': zone_data.get('name', 'Zone'),
                'color': zone_data.get('color')
            }
            
            # Add widget to container
            self.zones_container.addWidget(zone_widget)
            
            # Update zone counter
            self.zone_counter += 1
        
        # Reassign junctions and roads based on all saved areas
        # Process zones in order to respect assignment rules
        for zone_id, zone_data in saved_zones.items():
            zone_areas = zone_data.get('areas', [])
            for rect_data in zone_areas:
                # Convert list to QRectF if needed
                if isinstance(rect_data, list) and len(rect_data) == 4:
                    rect = QRectF(*rect_data)
                elif isinstance(rect_data, QRectF):
                    rect = rect_data
                else:
                    continue
                self._assign_junctions_and_roads_for_area(zone_id, rect)
        
        # Update counts for all loaded zones
        for zone_id in saved_zones.keys():
            self.update_zone_counts(zone_id)
        
        # Update statistics
        if saved_zones:
            self.update_statistics()
    
    def save_zones(self):
        """Save current zones to project configuration."""
        # Prepare zones data for saving
        zones_to_save = {}
        for zone_id, zone_data in self.zones.items():
            # Get areas from map_view (source of truth)
            areas = self.map_view.zones.get(zone_id, {}).get('areas', [])
            zones_to_save[zone_id] = {
                'name': zone_data.get('name', ''),
                'color': zone_data.get('color'),
                'areas': areas
            }
        
        self.config_manager.save_zones(zones_to_save)
    
    def add_porto_conversion_section(self, parent_layout):
        """Add Porto dataset conversion UI section (only called if Porto mode enabled)."""
        porto_group = QGroupBox("Porto Dataset Conversion")
        porto_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2e7d32;
            }
        """)
        porto_layout = QVBoxLayout()
        porto_layout.setSpacing(10)
        porto_layout.setContentsMargins(10, 10, 10, 10)
        
        # Porto dataset file path
        dataset_path_layout = QHBoxLayout()
        dataset_path_label = QLabel("Dataset File:")
        dataset_path_label.setFont(QFont("Arial", 10))
        dataset_path_layout.addWidget(dataset_path_label)
        
        self.porto_dataset_path_label = QLabel(str(self.porto_dataset_path) if hasattr(self, 'porto_dataset_path') else "Not set")
        self.porto_dataset_path_label.setFont(QFont("Arial", 9))
        self.porto_dataset_path_label.setStyleSheet("color: #666; padding: 5px;")
        self.porto_dataset_path_label.setWordWrap(True)
        self.porto_dataset_path_label.setSizePolicy(QWidget.Expanding, QWidget.Preferred)
        dataset_path_layout.addWidget(self.porto_dataset_path_label, stretch=1)
        porto_layout.addLayout(dataset_path_layout)
        
        # Conversion status
        self.porto_status_label = QLabel("Ready to convert")
        self.porto_status_label.setFont(QFont("Arial", 9))
        self.porto_status_label.setStyleSheet("color: #666; padding: 5px;")
        porto_layout.addWidget(self.porto_status_label)
        
        # Convert button
        convert_btn = QPushButton("Convert Trajectories to Routes")
        convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        convert_btn.clicked.connect(self.convert_porto_trajectories)
        porto_layout.addWidget(convert_btn)
        
        # Output route file location
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Route File:")
        output_label.setFont(QFont("Arial", 10))
        output_layout.addWidget(output_label)
        
        # Get Porto config folder for output
        porto_config_folder = self.config_manager.get_porto_config_folder()
        if porto_config_folder:
            output_path = Path(porto_config_folder) / 'porto.rou.xml'
        else:
            # Fallback
            project_path_obj = Path(self.project_path).resolve()
            workspace_root = project_path_obj
            while workspace_root != workspace_root.parent:
                if (workspace_root / 'Porto').exists():
                    break
                workspace_root = workspace_root.parent
            if not (workspace_root / 'Porto').exists():
                workspace_root = Path('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester')
            output_path = workspace_root / 'Porto' / 'config' / 'porto.rou.xml'
        
        self.porto_output_path_label = QLabel(str(output_path))
        self.porto_output_path_label.setFont(QFont("Arial", 9))
        self.porto_output_path_label.setStyleSheet("color: #666; padding: 5px;")
        self.porto_output_path_label.setWordWrap(True)
        self.porto_output_path_label.setSizePolicy(QWidget.Expanding, QWidget.Preferred)
        output_layout.addWidget(self.porto_output_path_label, stretch=1)
        porto_layout.addLayout(output_layout)
        
        porto_group.setLayout(porto_layout)
        parent_layout.addWidget(porto_group)
    
    def convert_porto_trajectories(self):
        """Convert Porto CSV trajectories to SUMO routes."""
        if not hasattr(self, 'porto_dataset_path') or not self.porto_dataset_path.exists():
            QMessageBox.warning(self, "Error", "Porto dataset file not found.")
            return
        
        if not self.network_parser:
            QMessageBox.warning(self, "Error", "Network not loaded. Please load a SUMO network first.")
            return
        
        self.porto_status_label.setText("Converting trajectories...")
        self.porto_status_label.setStyleSheet("color: #2196F3; padding: 5px;")
        
        try:
            # TODO: Implement Porto trajectory conversion logic
            QMessageBox.information(self, "Info", "Porto trajectory conversion will be implemented here.")
            self.porto_status_label.setText("Conversion completed")
            self.porto_status_label.setStyleSheet("color: #4CAF50; padding: 5px;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert trajectories:\n{str(e)}")
            self.porto_status_label.setText(f"Error: {str(e)}")
            self.porto_status_label.setStyleSheet("color: #f44336; padding: 5px;")
