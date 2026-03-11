"""
Route Generation page for creating SUMO route files manually.
"""

import ast
import csv
import gzip
import json
import re
import string
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, Signal, QTime, QUrl
from PySide6.QtGui import (QBrush, QColor, QDesktopServices, QFont, QGuiApplication,
                           QPainter, QPen)
from PySide6.QtWidgets import (QFrame, QGraphicsView, QGroupBox, QHBoxLayout,
                               QCheckBox, QGridLayout,
                               QFileDialog, QFormLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox,
                               QDoubleSpinBox, QTimeEdit, QVBoxLayout, QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.route_xml_generator import RouteXMLGenerator
from src.utils.simulation_db_service import SimulationDBService
from src.utils.sumo_config_manager import SUMOConfigManager


class NoScrollSpinBox(QSpinBox):
    """Spin box that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoScrollTimeEdit(QTimeEdit):
    """Time edit that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


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
                        # Highlight displayed zones via stronger outline (no fill tint).
                        painter.setPen(QPen(zone_color, 3))
                        painter.setBrush(Qt.NoBrush)
                    else:
                        # Standard zone outline only (no background tint).
                        painter.setPen(QPen(zone_color, 2))
                        painter.setBrush(Qt.NoBrush)
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


class SimulationPrepWorker(QObject):
    """Background worker that runs step-2 preparation checks."""

    progress_changed = Signal(int, str)
    finished = Signal(bool, str, object)

    def __init__(self, project_name: str, project_path: str):
        super().__init__()
        self.project_name = project_name
        self.project_path = project_path
        self._cancel_requested = False

    def cancel(self):
        """Request cancellation (best-effort)."""
        self._cancel_requested = True

    def run(self):
        """Run Step A preparation into simulation DB."""
        try:
            service = SimulationDBService(self.project_name, self.project_path)
            summary = service.prepare_initial_snapshot(
                progress_cb=lambda pct, msg: self.progress_changed.emit(pct, msg),
                is_cancelled=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                self.finished.emit(False, "Preparation cancelled.", {})
            else:
                self.finished.emit(True, "Simulation DB generated successfully.", summary)
        except Exception as exc:
            self.finished.emit(False, str(exc), {})


class RouteGenerationPage(QWidget):
    """Page for generating SUMO route files manually."""
    
    back_clicked = Signal()
    run_simulation_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.config_manager = SUMOConfigManager(project_path)
        self.route_generation_settings = self.config_manager.load_route_generation_settings()
        self.simulation_config_path = Path(self.project_path) / "simulation.config.json"
        self.network_parser = None
        self.route_generator = None
        self.zones = {}  # Dict of {zone_id: {'name': str, 'color': QColor, 'widget': QWidget, 'areas': [QRectF]}}
        self.zone_counter = 0  # Counter for zone letters (A, B, C, ...)
        self.current_zone_id = None  # Currently active zone for selection
        self.zones_locked = False  # Applied after zones/widgets are created
        self._restore_zones_locked = bool(self.route_generation_settings.get('zones_locked', False))
        saved_displayed = self.route_generation_settings.get('displayed_zones', [])
        self.displayed_zones = set(saved_displayed) if isinstance(saved_displayed, list) else set()
        self.detected_zones = {}  # zone_name -> {'edges': set[str], 'nodes': set[str], 'color': QColor}
        self.detected_visible_zones = set()
        self.detected_zone_row_widgets = {}
        self.detected_landmarks = {}  # landmark_name -> {'edges': set[str], 'color': QColor}
        self.detected_visible_landmarks = set()
        self.detected_landmark_row_widgets = {}
        self._landmarks_seed = {}
        self._default_landmarks_seed = {
            "home_zones": [],
            "work_zones": [],
            "restaurants_zones": [],
            "visit_count": 1,
        }
        self._default_landmarks_initialized = False
        self.vehicle_type_entries = []
        self.zone_allocation_entries = {}
        self._zone_allocation_seed = {}
        self.weekday_schedule_entries = []
        self._suppress_auto_persist = False
        self._prep_thread = None
        self._prep_worker = None
        
        # Track assignments: junction_id -> zone_id, edge_id -> zone_id
        self.junction_assignments = {}  # Dict of {junction_id: zone_id}
        self.road_assignments = {}  # Dict of {edge_id: zone_id}
        
        self.init_ui()
        self.load_simulation_config_from_project()
        self.load_network()
        # Load saved zones after network is loaded (but before Porto neighborhoods)
        # This ensures saved zones are loaded first, then Porto neighborhoods are added if needed
        if hasattr(self, 'zones_container'):
            self.load_saved_zones()
            self._restore_route_generation_state()
    
    def init_ui(self):
        """Initialize the page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel(f"Simulation Definition - {self.project_name}")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        back_btn = QPushButton("Back")
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
        self.map_view.set_edge_thickness_multiplier(1.6)
        self.map_view.setMinimumSize(600, 500)
        self.map_view.area_selected.connect(self.on_area_selected)
        self.map_view.zone_area_selected.connect(self.on_zone_area_selected)
        self.map_view.network_render_finished.connect(self.on_map_network_render_finished)
        # Store reference to parent for zones_locked access
        self.map_view.route_page = self
        map_layout.addWidget(self.map_view)
        map_group.setLayout(map_layout)
        content_layout.addWidget(map_group, stretch=2)

        # Right side: Simulation Configuration (fixed title + scrollable form)
        config_panel = QGroupBox("Simulation Configuration")
        config_panel.setStyleSheet("""
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
        config_panel_layout = QVBoxLayout()
        config_panel_layout.setContentsMargins(8, 8, 8, 8)
        config_panel_layout.setSpacing(6)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        config_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        config_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        config_group = QWidget()
        config_layout = QVBoxLayout()
        config_layout.setContentsMargins(12, 12, 12, 12)
        config_layout.setSpacing(10)

        # Detected zones block
        detected_group = QGroupBox("Zones (Neighborhoods)")
        self._set_config_block_style(detected_group, "#dceeff")
        detected_layout = QVBoxLayout()
        detected_layout.setSpacing(8)

        detected_top_row = QHBoxLayout()
        self.detect_zones_btn = QPushButton("Update Zones")
        self.detect_zones_btn.setToolTip(
            "A zone is an attribute attached to both edges and nodes.\n"
            "Detect Zones scans the loaded network file, detects zones, counts\n"
            "their edges/nodes, and lets you show/hide each zone on the map."
        )
        self.detect_zones_btn.clicked.connect(self.detect_zones_from_network)
        detected_top_row.addWidget(self.detect_zones_btn)
        detected_top_row.addStretch()
        detected_layout.addLayout(detected_top_row)

        self.detected_zones_list_layout = QVBoxLayout()
        self.detected_zones_list_layout.setSpacing(4)
        detected_layout.addLayout(self.detected_zones_list_layout)

        self.detected_zones_status = QLabel("No zones detected yet.")
        self.detected_zones_status.setStyleSheet("color: #666;")
        detected_layout.addWidget(self.detected_zones_status)

        detected_group.setLayout(detected_layout)
        config_layout.addWidget(detected_group)

        # Vehicle types block
        vehicle_types_group = QGroupBox("Vehicle Types")
        self._set_config_block_style(vehicle_types_group, "#e2f7db")
        vehicle_types_layout = QVBoxLayout()
        vehicle_types_layout.setSpacing(8)

        self.vehicle_types_cards_layout = QVBoxLayout()
        self.vehicle_types_cards_layout.setSpacing(6)
        vehicle_types_layout.addLayout(self.vehicle_types_cards_layout)

        add_type_row = QHBoxLayout()
        add_type_btn = QPushButton("+")
        add_type_btn.setFixedWidth(28)
        add_type_btn.setToolTip("Add vehicle type")
        add_type_btn.clicked.connect(self.add_vehicle_type_card)
        add_type_row.addWidget(add_type_btn)
        add_type_row.addStretch()
        vehicle_types_layout.addLayout(add_type_row)

        vehicle_types_group.setLayout(vehicle_types_layout)
        config_layout.addWidget(vehicle_types_group)

        # Default vehicle types
        self.add_vehicle_type_card("passenger", 4.5, 1.8, 1.5, 33.33, "blue")
        self.add_vehicle_type_card("bus", 12.0, 2.5, 3.2, 22.22, "yellow")
        self.add_vehicle_type_card("truck", 8.0, 2.5, 3.0, 25.0, "red")

        # Zone vehicle allocation block (JSON-structured UI)
        zone_alloc_group = QGroupBox("Zone Vehicle Allocation")
        self._set_config_block_style(zone_alloc_group, "#ffe9cc")
        zone_alloc_layout = QVBoxLayout()
        zone_alloc_layout.setSpacing(8)
        self.zone_allocation_total_label = QLabel("Total: 0%")
        self.zone_allocation_total_label.setStyleSheet("color: #666; font-size: 10px;")
        zone_alloc_layout.addWidget(self.zone_allocation_total_label)
        self.zone_allocation_cards_layout = QVBoxLayout()
        self.zone_allocation_cards_layout.setSpacing(6)
        zone_alloc_layout.addLayout(self.zone_allocation_cards_layout)
        zone_alloc_group.setLayout(zone_alloc_layout)
        config_layout.addWidget(zone_alloc_group)
        self.refresh_zone_allocation_section()

        # Core simulation fields
        core_group = QGroupBox("Core Settings")
        self._set_config_block_style(core_group, "#ece6ff")
        core_form = QFormLayout()

        snapshot_row = QHBoxLayout()
        self.snapshot_dir_input = QLineEdit(str((Path(self.project_path) / "snapshots").resolve()))
        browse_snapshot_btn = QPushButton("Browse")
        browse_snapshot_btn.clicked.connect(self.browse_snapshot_dir)
        snapshot_row.addWidget(self.snapshot_dir_input)
        snapshot_row.addWidget(browse_snapshot_btn)
        core_form.addRow("snapshot_dir", snapshot_row)

        self.snapshot_interval_spin = NoScrollSpinBox()
        self.snapshot_interval_spin.setRange(1, 3600)
        self.snapshot_interval_spin.setValue(30)
        core_form.addRow("snapshot_interval_sec", self.snapshot_interval_spin)

        self.simulation_weeks_spin = NoScrollSpinBox()
        self.simulation_weeks_spin.setRange(1, 520)
        self.simulation_weeks_spin.setValue(1)
        core_form.addRow("simulation_weeks", self.simulation_weeks_spin)

        self.total_vehicles_spin = NoScrollSpinBox()
        self.total_vehicles_spin.setRange(1, 10_000_000)
        self.total_vehicles_spin.setValue(200_000)
        core_form.addRow("total_num_vehicles", self.total_vehicles_spin)

        self.dev_fraction_spin = NoScrollDoubleSpinBox()
        self.dev_fraction_spin.setRange(0.0, 1.0)
        self.dev_fraction_spin.setDecimals(3)
        self.dev_fraction_spin.setSingleStep(0.05)
        self.dev_fraction_spin.setValue(1.0)
        core_form.addRow("dev_fraction", self.dev_fraction_spin)
        core_group.setLayout(core_form)
        config_layout.addWidget(core_group)

        # Landmarks block (from net.xml edge params)
        landmarks_group = QGroupBox("Landmarks")
        self._set_config_block_style(landmarks_group, "#d7f6ee")
        landmarks_layout = QVBoxLayout()
        landmarks_layout.setSpacing(8)

        landmarks_top_row = QHBoxLayout()
        self.update_landmarks_btn = QPushButton("Update Landmarks")
        self.update_landmarks_btn.setToolTip(
            "Scan the loaded network file for edge params:\n"
            "<param key=\"landmark\" value=\"<name>\"/>"
        )
        self.update_landmarks_btn.clicked.connect(self.detect_landmarks_from_network)
        landmarks_top_row.addWidget(self.update_landmarks_btn)
        landmarks_top_row.addStretch()
        landmarks_layout.addLayout(landmarks_top_row)

        self.detected_landmarks_list_layout = QVBoxLayout()
        self.detected_landmarks_list_layout.setSpacing(4)
        landmarks_layout.addLayout(self.detected_landmarks_list_layout)

        self.detected_landmarks_status = QLabel("No landmarks detected yet.")
        self.detected_landmarks_status.setStyleSheet("color: #666;")
        landmarks_layout.addWidget(self.detected_landmarks_status)

        # Default landmarks subsection
        default_landmarks_group = QGroupBox("Default Landmarks")
        self._set_config_block_style(default_landmarks_group, "#cdeee3")
        default_landmarks_form = QFormLayout()
        default_landmarks_form.setSpacing(6)

        self.default_home_zones_list = self._create_checkbox_group_widget()
        default_landmarks_form.addRow("Home", self.default_home_zones_list)

        self.default_work_zones_list = self._create_checkbox_group_widget()
        default_landmarks_form.addRow("Work", self.default_work_zones_list)

        self.default_restaurants_zones_list = self._create_checkbox_group_widget()
        default_landmarks_form.addRow("Restaurants", self.default_restaurants_zones_list)

        self.default_visit_spin = NoScrollSpinBox()
        self.default_visit_spin.setRange(1, 3)
        self.default_visit_spin.setValue(1)
        self.default_visit_spin.valueChanged.connect(lambda _: self._persist_default_landmarks_only())
        default_landmarks_form.addRow("Visit", self.default_visit_spin)

        default_landmarks_group.setLayout(default_landmarks_form)
        landmarks_layout.addWidget(default_landmarks_group)

        landmarks_group.setLayout(landmarks_layout)
        config_layout.addWidget(landmarks_group)

        # Weekday schedule block (structured UI)
        weekday_group = QGroupBox("Weekday Schedule")
        self._set_config_block_style(weekday_group, "#ffdce8")
        weekday_layout = QVBoxLayout()
        weekday_layout.setSpacing(8)
        self.weekday_schedule_cards_layout = QVBoxLayout()
        self.weekday_schedule_cards_layout.setSpacing(6)
        weekday_layout.addLayout(self.weekday_schedule_cards_layout)
        weekday_add_row = QHBoxLayout()
        self.add_weekday_schedule_btn = QPushButton("+")
        self.add_weekday_schedule_btn.setFixedWidth(28)
        self.add_weekday_schedule_btn.setToolTip("Add weekday schedule subsection")
        self.add_weekday_schedule_btn.clicked.connect(self.add_weekday_schedule_card)
        weekday_add_row.addWidget(self.add_weekday_schedule_btn)
        weekday_add_row.addStretch()
        weekday_layout.addLayout(weekday_add_row)
        weekday_group.setLayout(weekday_layout)
        config_layout.addWidget(weekday_group)

        config_actions_row = QHBoxLayout()
        config_actions_row.addStretch()
        self.generate_sim_db_btn = QPushButton("Generate Simulation DB")
        self.generate_sim_db_btn.clicked.connect(self.on_generate_simulation_db_clicked)
        config_actions_row.addWidget(self.generate_sim_db_btn)
        self.cancel_prepare_btn = QPushButton("Cancel")
        self.cancel_prepare_btn.setEnabled(False)
        self.cancel_prepare_btn.clicked.connect(self.on_cancel_preparation_clicked)
        config_actions_row.addWidget(self.cancel_prepare_btn)
        open_config_btn = QPushButton("Open simulation.config.json")
        open_config_btn.clicked.connect(self.open_simulation_config_json)
        config_actions_row.addWidget(open_config_btn)
        run_sim_btn = QPushButton("Run Simulation")
        run_sim_btn.clicked.connect(self.run_simulation_clicked.emit)
        config_actions_row.addWidget(run_sim_btn)
        config_layout.addLayout(config_actions_row)
        self.prepare_progress_bar = QProgressBar()
        self.prepare_progress_bar.setRange(0, 100)
        self.prepare_progress_bar.setValue(0)
        self.prepare_progress_bar.setVisible(False)
        config_layout.addWidget(self.prepare_progress_bar)
        self.prepare_status_label = QLabel("")
        self.prepare_status_label.setStyleSheet("color: #666;")
        self.prepare_status_label.setVisible(False)
        config_layout.addWidget(self.prepare_status_label)

        # Keep stats label available for existing logic.
        self.stats_label = QLabel("")
        self.stats_label.setVisible(False)
        config_layout.addWidget(self.stats_label)
        config_layout.addStretch()

        config_group.setLayout(config_layout)
        config_scroll.setWidget(config_group)
        config_panel_layout.addWidget(config_scroll)
        config_panel.setLayout(config_panel_layout)
        content_layout.addWidget(config_panel, stretch=1)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        self._connect_auto_persist_signals()
    
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
                        # Auto-detect zones from network attributes on load.
                        self.detect_zones_from_network()
                        # Auto-detect landmarks from edge params on load.
                        self.detect_landmarks_from_network()
                        # Update statistics when network is loaded
                        self.update_statistics()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load network: {str(e)}")

    def browse_snapshot_dir(self):
        """Select snapshot output directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Snapshot Directory",
            self.snapshot_dir_input.text().strip() or self.project_path
        )
        if folder:
            self.snapshot_dir_input.setText(folder)
            self._persist_current_config_safely()

    def _set_config_block_style(self, group: QGroupBox, background_hex: str):
        """Apply only light background color (minimal-risk styling)."""
        group.setAttribute(Qt.WA_StyledBackground, True)
        group.setStyleSheet(
            f"QGroupBox {{ background-color: {background_hex}; border: 1px solid #d5d5d5; border-radius: 6px; }}"
        )

    def open_simulation_config_json(self):
        """Open project simulation.config.json using the default system app."""
        try:
            self._persist_current_config_safely()
            self.simulation_config_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.simulation_config_path.exists():
                self._write_project_simulation_config(self._build_config_from_form())
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.simulation_config_path)))
            if not opened:
                QMessageBox.warning(self, "Open Failed", f"Could not open:\n{self.simulation_config_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", f"Could not open config JSON:\n{exc}")

    def on_generate_simulation_db_clicked(self):
        """Run step-2 preparation checks in background."""
        if self._prep_thread is not None:
            return
        self.generate_sim_db_btn.setEnabled(False)
        self.cancel_prepare_btn.setEnabled(True)
        self.prepare_progress_bar.setVisible(True)
        self.prepare_status_label.setVisible(True)
        self.prepare_progress_bar.setValue(0)
        self.prepare_status_label.setText("Starting preparation...")
        self.prepare_status_label.setStyleSheet("color: #666;")

        self._prep_thread = QThread(self)
        self._prep_worker = SimulationPrepWorker(self.project_name, self.project_path)
        self._prep_worker.moveToThread(self._prep_thread)
        self._prep_thread.started.connect(self._prep_worker.run)
        self._prep_worker.progress_changed.connect(self.on_preparation_progress_changed)
        self._prep_worker.finished.connect(self.on_preparation_finished)
        self._prep_worker.finished.connect(self._prep_thread.quit)
        self._prep_thread.finished.connect(self._prep_thread.deleteLater)
        self._prep_thread.start()

    def on_cancel_preparation_clicked(self):
        """Best-effort cancellation for running preparation worker."""
        if self._prep_worker is not None:
            self._prep_worker.cancel()
            self.prepare_status_label.setText("Cancelling...")
            self.prepare_status_label.setStyleSheet("color: #c62828;")
            self.cancel_prepare_btn.setEnabled(False)

    def on_preparation_progress_changed(self, value: int, message: str):
        """Update progress bar and status text."""
        self.prepare_progress_bar.setValue(max(0, min(100, int(value))))
        self.prepare_status_label.setText(message)

    def on_preparation_finished(self, success: bool, message: str, summary: object):
        """Handle completion of step-2 background preparation checks."""
        self.generate_sim_db_btn.setEnabled(True)
        self.cancel_prepare_btn.setEnabled(False)
        if success:
            details = ""
            if isinstance(summary, dict):
                details = (
                    f" db={summary.get('db_path', '')},"
                    f" zones={summary.get('zone_count', 0)},"
                    f" vehicles={summary.get('vehicle_count', 0)}"
                )
            self.prepare_status_label.setText(f"{message}{details}")
            self.prepare_status_label.setStyleSheet("color: #2e7d32;")
            self.prepare_progress_bar.setValue(100)
        else:
            self.prepare_status_label.setText(f"Preparation failed: {message}")
            self.prepare_status_label.setStyleSheet("color: #c62828;")

        if self._prep_worker is not None:
            self._prep_worker.deleteLater()
        self._prep_worker = None
        self._prep_thread = None

    def _connect_auto_persist_signals(self):
        """Persist to project simulation config whenever form values change."""
        self.snapshot_dir_input.textChanged.connect(lambda _: self._persist_current_config_safely())
        self.snapshot_interval_spin.valueChanged.connect(lambda _: self._persist_current_config_safely())
        self.simulation_weeks_spin.valueChanged.connect(lambda _: self._persist_current_config_safely())
        self.total_vehicles_spin.valueChanged.connect(lambda _: self._persist_current_config_safely())
        self.dev_fraction_spin.valueChanged.connect(lambda _: self._persist_current_config_safely())

    def _read_project_simulation_config(self) -> Dict:
        """Read simulation config from project's simulation.config.json."""
        if not self.simulation_config_path.exists():
            return {}
        try:
            with open(self.simulation_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_project_simulation_config(self, config: Dict):
        """Write simulation config to project's simulation.config.json."""
        self.simulation_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.simulation_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_simulation_config_from_project(self):
        """Load form values from project's simulation.config.json if present."""
        config = self._read_project_simulation_config()
        self._suppress_auto_persist = True
        try:
            if not config:
                # First-time defaults required by user: noise=100, passenger=100.
                self._zone_allocation_seed = {"noise": {"percentage": 100.0}}
                self.refresh_zone_allocation_section(preserve_existing=False)
                return

            self.snapshot_dir_input.setText(config.get("snapshot_dir", self.snapshot_dir_input.text()))
            if "snapshot_interval_sec" in config:
                self.snapshot_interval_spin.setValue(int(config.get("snapshot_interval_sec", 30)))

            vg = config.get("vehicle_generation", {})
            if isinstance(vg, dict):
                if "simulation_weeks" in vg:
                    self.simulation_weeks_spin.setValue(int(vg.get("simulation_weeks", 1)))
                if "total_num_vehicles" in vg:
                    self.total_vehicles_spin.setValue(int(vg.get("total_num_vehicles", 200000)))
                if "dev_fraction" in vg:
                    self.dev_fraction_spin.setValue(float(vg.get("dev_fraction", 1.0)))

                loaded_types = vg.get("vehicle_types", {})
                if isinstance(loaded_types, dict) and loaded_types:
                    # Replace defaults with loaded vehicle types.
                    while self.vehicle_type_entries:
                        self.remove_vehicle_type_card(self.vehicle_type_entries[0]["container"])
                    for type_name, attrs in loaded_types.items():
                        attrs = attrs if isinstance(attrs, dict) else {}
                        self.add_vehicle_type_card(
                            name=type_name,
                            length=float(attrs.get("length", 4.5)),
                            width=float(attrs.get("width", 1.8)),
                            height=float(attrs.get("height", 1.5)),
                            max_speed=float(attrs.get("max_speed", 30.0)),
                            color=str(attrs.get("color", "blue")),
                        )

                loaded_alloc = vg.get("zone_allocation", {})
                if isinstance(loaded_alloc, dict):
                    self._zone_allocation_seed = loaded_alloc

            loaded_landmarks = config.get("landmarks", {})
            if isinstance(loaded_landmarks, dict):
                default_landmarks = loaded_landmarks.get("default_landmarks", {})
                if not isinstance(default_landmarks, dict):
                    default_landmarks = config.get("default_landmarks", {})
                visit_count = 1
                if isinstance(default_landmarks, dict):
                    try:
                        visit_count = int(default_landmarks.get("visit_count", 1))
                    except (TypeError, ValueError):
                        visit_count = 1
                self._default_landmarks_seed = {
                    "home_zones": list(default_landmarks.get("home_zones", [])) if isinstance(default_landmarks, dict) else [],
                    "work_zones": list(default_landmarks.get("work_zones", [])) if isinstance(default_landmarks, dict) else [],
                    "restaurants_zones": list(default_landmarks.get("restaurants_zones", [])) if isinstance(default_landmarks, dict) else [],
                    "visit_count": visit_count,
                }
                self._default_landmarks_initialized = False
                self._landmarks_seed = {
                    name: edges for name, edges in loaded_landmarks.items()
                    if name != "default_landmarks" and isinstance(edges, list)
                }
                self._set_landmarks_from_mapping(self._landmarks_seed)
            self.refresh_zone_allocation_section(preserve_existing=False)
            self.refresh_default_landmark_zone_options()
            self.load_weekday_schedule_from_config(config.get("weekday_schedule", []))
        finally:
            self._suppress_auto_persist = False
        self._persist_current_config_safely()

    def _extract_zone_name_from_id(self, entity_id: str) -> Optional[str]:
        """Extract a zone name from an edge/node id using a simple prefix heuristic."""
        if not entity_id:
            return None
        match = re.search(r"[A-Za-z]", entity_id)
        if not match:
            return None
        return match.group(0).upper()

    def _load_zone_attributes_from_network_file(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load explicit zone attributes from net file for edges and nodes/junctions."""
        edge_zone_map: Dict[str, str] = {}
        node_zone_map: Dict[str, str] = {}
        if not self.network_parser or not getattr(self.network_parser, "net_file", None):
            return edge_zone_map, node_zone_map

        net_path = Path(self.network_parser.net_file)
        if not net_path.exists():
            return edge_zone_map, node_zone_map

        try:
            if net_path.suffix == '.gz' or net_path.name.endswith('.gz'):
                with gzip.open(net_path, 'rb') as f:
                    tree = ET.parse(f)
            else:
                tree = ET.parse(net_path)
            root = tree.getroot()
        except Exception:
            return edge_zone_map, node_zone_map

        def get_zone_value(elem) -> Optional[str]:
            zone_attr = elem.get('zone')
            if zone_attr:
                return zone_attr.strip()
            for param in elem.findall('param'):
                if (param.get('key') or '').strip().lower() == 'zone':
                    value = (param.get('value') or '').strip()
                    if value:
                        return value
            return None

        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            if not edge_id:
                continue
            zone_value = get_zone_value(edge)
            if zone_value:
                edge_zone_map[edge_id] = zone_value

        for node in root.findall('node'):
            node_id = node.get('id')
            if not node_id:
                continue
            zone_value = get_zone_value(node)
            if zone_value:
                node_zone_map[node_id] = zone_value

        for junction in root.findall('junction'):
            node_id = junction.get('id')
            if not node_id:
                continue
            zone_value = get_zone_value(junction)
            if zone_value:
                node_zone_map[node_id] = zone_value

        return edge_zone_map, node_zone_map

    def _load_landmarks_from_network_file(self) -> Dict[str, set]:
        """Load landmarks from edge params: <param key="landmark" value="<name>"/>."""
        landmarks_map: Dict[str, set] = {}
        if not self.network_parser or not getattr(self.network_parser, "net_file", None):
            return landmarks_map

        net_path = Path(self.network_parser.net_file)
        if not net_path.exists():
            return landmarks_map

        try:
            if net_path.suffix == '.gz' or net_path.name.endswith('.gz'):
                with gzip.open(net_path, 'rb') as f:
                    tree = ET.parse(f)
            else:
                tree = ET.parse(net_path)
            root = tree.getroot()
        except Exception:
            return landmarks_map

        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            if not edge_id:
                continue
            for param in edge.findall('param'):
                if (param.get('key') or '').strip().lower() != 'landmark':
                    continue
                landmark_name = (param.get('value') or '').strip()
                if not landmark_name:
                    continue
                landmarks_map.setdefault(landmark_name, set()).add(edge_id)
        return landmarks_map

    def _clear_detected_landmark_rows(self):
        """Remove detected-landmark row widgets from the UI."""
        while self.detected_landmarks_list_layout.count():
            item = self.detected_landmarks_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.detected_landmark_row_widgets = {}

    def _set_landmarks_from_mapping(self, landmarks_mapping: Dict):
        """Set detected landmarks from mapping name -> iterable(edge_ids)."""
        self.detected_landmarks = {}
        self.detected_visible_landmarks = set()
        if hasattr(self, "detected_landmarks_list_layout"):
            self._clear_detected_landmark_rows()

        palette = [
            QColor(255, 105, 180),  # hot pink
            QColor(255, 165, 0),    # orange
            QColor(138, 43, 226),   # blue violet
            QColor(0, 191, 255),    # deep sky blue
            QColor(60, 179, 113),   # medium sea green
            QColor(255, 69, 0),     # orange red
            QColor(70, 130, 180),   # steel blue
            QColor(205, 92, 92),    # indian red
        ]

        for idx, landmark_name in enumerate(sorted(landmarks_mapping.keys())):
            edge_ids_raw = landmarks_mapping.get(landmark_name, [])
            edge_ids = set(edge_ids_raw) if edge_ids_raw is not None else set()
            color = palette[idx % len(palette)]
            self.detected_landmarks[landmark_name] = {
                "edges": edge_ids,
                "color": color,
            }
            if hasattr(self, "detected_landmarks_list_layout"):
                row = self._create_detected_landmark_row(landmark_name, len(edge_ids))
                self.detected_landmarks_list_layout.addWidget(row)

        if hasattr(self, "detected_landmarks_status"):
            if self.detected_landmarks:
                self.detected_landmarks_status.setText(f"Detected {len(self.detected_landmarks)} landmark(s).")
            else:
                self.detected_landmarks_status.setText("No landmarks detected.")

    def _get_zones_with_positive_allocation(self) -> List[str]:
        """Return zone names that currently have percentage > 0 (excluding noise)."""
        zones = []
        for zone_name, entry in self.zone_allocation_entries.items():
            if zone_name.lower() == "noise":
                continue
            if entry["percentage_spin"].value() > 0:
                zones.append(zone_name)
        return sorted(zones)

    def _get_zones_with_positive_allocation_from_seed(self) -> List[str]:
        """Return positive-allocation zones from seed data when UI cards are not ready."""
        seed = self._zone_allocation_seed if isinstance(self._zone_allocation_seed, dict) else {}
        zones = []
        for zone_name, payload in seed.items():
            if str(zone_name).lower() == "noise":
                continue
            if not isinstance(payload, dict):
                continue
            try:
                pct = float(payload.get("percentage", 0.0))
            except (TypeError, ValueError):
                pct = 0.0
            if pct > 0.0:
                zones.append(str(zone_name))
        return sorted(zones)

    def _get_positive_zone_names_for_schedule_rules(self) -> List[str]:
        """Return positive zones using UI values when available, otherwise seed values."""
        zones = self._get_zones_with_positive_allocation()
        if zones:
            return zones
        return self._get_zones_with_positive_allocation_from_seed()

    def _expand_schedule_landmark_aliases(self, selected_landmarks: List[str]) -> List[str]:
        """Expand alias selections into concrete landmark names for config output."""
        expanded = []
        seen = set()
        positive_zones = self._get_positive_zone_names_for_schedule_rules()
        visit_count = int(self.default_visit_spin.value()) if hasattr(self, "default_visit_spin") else int(self._default_landmarks_seed.get("visit_count", 1))
        visit_count = max(1, min(3, visit_count))
        for name in selected_landmarks:
            key = str(name).strip()
            if not key:
                continue
            if key == "restaurants":
                for zone_name in positive_zones:
                    restaurant_name = f"restaurant{zone_name}"
                    if restaurant_name not in seen:
                        seen.add(restaurant_name)
                        expanded.append(restaurant_name)
                continue
            if key == "visit":
                for idx in range(1, visit_count + 1):
                    visit_name = f"visit{idx}"
                    if visit_name not in seen:
                        seen.add(visit_name)
                        expanded.append(visit_name)
                continue
            if key not in seen:
                seen.add(key)
                expanded.append(key)
        return expanded

    def _normalize_schedule_landmark_aliases_for_ui(self, selected_landmarks: List[str]) -> List[str]:
        """Normalize concrete landmark names from config to UI alias checkboxes."""
        normalized = []
        seen = set()
        positive_zones = self._get_positive_zone_names_for_schedule_rules()
        restaurant_aliases = {f"restaurant{zone_name}" for zone_name in positive_zones}
        has_restaurants_alias = False
        has_visit_alias = False
        for raw_name in selected_landmarks:
            name = str(raw_name).strip()
            if not name:
                continue
            lower_name = name.lower()
            if name in restaurant_aliases:
                has_restaurants_alias = True
                continue
            if re.match(r"^visit\d+$", lower_name):
                has_visit_alias = True
                continue
            if name not in seen:
                seen.add(name)
                normalized.append(name)
        if has_restaurants_alias and "restaurants" not in seen:
            normalized.append("restaurants")
            seen.add("restaurants")
        if has_visit_alias and "visit" not in seen:
            normalized.append("visit")
            seen.add("visit")
        return normalized

    def _create_checkbox_group_widget(self) -> QWidget:
        """Create a checkbox container with up to 4 options per row."""
        container = QWidget()
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        for col in range(4):
            layout.setColumnStretch(col, 1)
        container.setLayout(layout)
        container._checkboxes = {}  # type: ignore[attr-defined]
        return container

    def _clear_layout(self, layout):
        """Remove all items/widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _get_selected_zone_names_from_list(self, checkbox_group_widget: QWidget) -> List[str]:
        """Return checked values from a checkbox group widget."""
        checkboxes = getattr(checkbox_group_widget, "_checkboxes", {})
        return [name for name, cb in checkboxes.items() if cb.isChecked()]

    def _collect_default_landmarks_from_form(self) -> Dict[str, object]:
        """Collect only default_landmarks payload from current form fields."""
        if not hasattr(self, "default_home_zones_list"):
            return {
                "home_zones": [],
                "work_zones": [],
                "restaurants_zones": [],
                "visit_count": 1,
            }
        return {
            "home_zones": self._get_selected_zone_names_from_list(self.default_home_zones_list),
            "work_zones": self._get_selected_zone_names_from_list(self.default_work_zones_list),
            "restaurants_zones": self._get_selected_zone_names_from_list(self.default_restaurants_zones_list),
            "visit_count": int(self.default_visit_spin.value()) if hasattr(self, "default_visit_spin") else 1,
        }

    def _set_checkbox_group_options(
        self,
        checkbox_group_widget: QWidget,
        options: List[str],
        selected_names: List[str],
        on_change=None,
    ):
        """Set checkbox options preserving checked values."""
        layout = checkbox_group_widget.layout()
        if layout is None:
            layout = QGridLayout()
            checkbox_group_widget.setLayout(layout)
        self._clear_layout(layout)
        selected_set = set(selected_names)
        checkboxes = {}
        for idx, option_name in enumerate(options):
            checkbox = QCheckBox(str(option_name))
            checkbox.setChecked(option_name in selected_set)
            checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if on_change is not None:
                checkbox.toggled.connect(on_change)
            row = idx // 4
            col = idx % 4
            layout.addWidget(checkbox, row, col)
            checkboxes[option_name] = checkbox
        if isinstance(layout, QGridLayout):
            for col in range(4):
                layout.setColumnStretch(col, 1)
        checkbox_group_widget._checkboxes = checkboxes  # type: ignore[attr-defined]

    def refresh_default_landmark_zone_options(self):
        """Refresh Home/Work/Restaurants selectable zones from allocation > 0."""
        if not hasattr(self, "default_home_zones_list"):
            return

        if self._default_landmarks_initialized:
            existing_home = self._get_selected_zone_names_from_list(self.default_home_zones_list)
            existing_work = self._get_selected_zone_names_from_list(self.default_work_zones_list)
            existing_restaurants = self._get_selected_zone_names_from_list(self.default_restaurants_zones_list)
            visit_value = int(self.default_visit_spin.value())
        else:
            existing_home = list(self._default_landmarks_seed.get("home_zones", []))
            existing_work = list(self._default_landmarks_seed.get("work_zones", []))
            existing_restaurants = list(self._default_landmarks_seed.get("restaurants_zones", []))
            visit_value = int(self._default_landmarks_seed.get("visit_count", 1))

        options = self._get_zones_with_positive_allocation()
        widgets = [
            (self.default_home_zones_list, existing_home),
            (self.default_work_zones_list, existing_work),
            (self.default_restaurants_zones_list, existing_restaurants),
        ]
        for checkbox_group_widget, selected_values in widgets:
            self._set_checkbox_group_options(
                checkbox_group_widget,
                options,
                selected_values,
                on_change=lambda _checked: self._persist_default_landmarks_only(),
            )

        self.default_visit_spin.blockSignals(True)
        self.default_visit_spin.setValue(max(1, min(3, visit_value)))
        self.default_visit_spin.blockSignals(False)
        self._default_landmarks_initialized = True

    def _persist_default_landmarks_only(self):
        """Persist default_landmarks even if other form sections are currently invalid."""
        if self._suppress_auto_persist:
            return
        try:
            config = self._read_project_simulation_config()
            if not isinstance(config, dict):
                config = {}
            landmarks = config.get("landmarks", {})
            if not isinstance(landmarks, dict):
                landmarks = {}
            landmarks["default_landmarks"] = self._collect_default_landmarks_from_form()
            config["landmarks"] = landmarks
            self._write_project_simulation_config(config)
        except Exception:
            pass

    def _get_landmark_option_names(self) -> List[str]:
        """Return selectable landmark names, including default landmarks."""
        names = set()
        names.update(self.detected_landmarks.keys())
        if isinstance(self._landmarks_seed, dict):
            names.update([k for k in self._landmarks_seed.keys() if k != "default_landmarks"])
        names.update(["home", "work", "restaurants", "visit"])
        return sorted(names)

    def _sync_multiselect_list(self, checkbox_group_widget: QWidget, options: List[str], selected_values: List[str]):
        """Replace checkbox options and preserve valid checked values."""
        self._set_checkbox_group_options(
            checkbox_group_widget,
            options,
            selected_values,
            on_change=lambda _checked: self._persist_current_config_safely(),
        )

    def _clear_weekday_schedule_cards(self):
        """Remove all weekday schedule card widgets."""
        while self.weekday_schedule_cards_layout.count():
            item = self.weekday_schedule_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.weekday_schedule_entries = []

    def _apply_weekday_schedule_card_colors(self):
        """Apply alternating background colors to weekday schedule cards."""
        palette = [
            "#ffe8ef",  # pink
            "#e8f1ff",  # blue
            "#e9fbe7",  # green
            "#fff4dc",  # peach
            "#efe8ff",  # purple
            "#e6fbf7",  # mint
        ]
        for idx, entry in enumerate(self.weekday_schedule_entries):
            card = entry.get("widget")
            if card is None:
                continue
            color = palette[idx % len(palette)]
            card.setAttribute(Qt.WA_StyledBackground, True)
            card.setStyleSheet(
                f"QGroupBox {{ background-color: {color}; border: 1px solid #d5d5d5; border-radius: 6px; }}"
            )

    def refresh_weekday_schedule_options(self):
        """Refresh source_zones/origin/destination options based on current data."""
        if not hasattr(self, "weekday_schedule_entries"):
            return
        zone_options = self._get_zones_with_positive_allocation()
        landmark_options = self._get_landmark_option_names()
        for entry in self.weekday_schedule_entries:
            selected_source = self._get_selected_zone_names_from_list(entry["source_zones"])
            selected_origin = self._get_selected_zone_names_from_list(entry["origin"])
            selected_destination = self._get_selected_zone_names_from_list(entry["destination"])
            self._sync_multiselect_list(entry["source_zones"], zone_options, selected_source)
            self._sync_multiselect_list(entry["origin"], landmark_options, selected_origin)
            self._sync_multiselect_list(entry["destination"], landmark_options, selected_destination)

    def add_weekday_schedule_card(self, seed: Optional[Dict] = None):
        """Add one weekday schedule subsection card."""
        seed = seed if isinstance(seed, dict) else {}
        card = QGroupBox("Schedule")
        card_layout = QVBoxLayout()
        card_layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("name"))
        name_input = QLineEdit(str(seed.get("name", "")).strip())
        top_row.addWidget(name_input)
        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(28)
        top_row.addWidget(remove_btn)
        card_layout.addLayout(top_row)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("start_time"))
        start_time = NoScrollTimeEdit()
        start_time.setDisplayFormat("HH:mm")
        start_time.setTime(QTime.fromString(str(seed.get("start_time", "08:00")), "HH:mm"))
        if not start_time.time().isValid():
            start_time.setTime(QTime(8, 0))
        time_row.addWidget(start_time)
        time_row.addWidget(QLabel("end_time"))
        end_time = NoScrollTimeEdit()
        end_time.setDisplayFormat("HH:mm")
        end_time.setTime(QTime.fromString(str(seed.get("end_time", "09:00")), "HH:mm"))
        if not end_time.time().isValid():
            end_time.setTime(QTime(9, 0))
        time_row.addWidget(end_time)
        time_row.addStretch()
        card_layout.addLayout(time_row)

        repeat_days_list = self._create_checkbox_group_widget()
        selected_days = set(seed.get("repeat_on_days", [])) if isinstance(seed.get("repeat_on_days", []), list) else set()
        self._set_checkbox_group_options(
            repeat_days_list,
            [str(day) for day in range(1, 8)],
            [str(day) for day in selected_days],
            on_change=lambda _checked: self._persist_current_config_safely(),
        )
        card_layout.addWidget(QLabel("repeat_on_days"))
        card_layout.addWidget(repeat_days_list)

        vpm_row = QHBoxLayout()
        vpm_row.addWidget(QLabel("vpm_rate"))
        vpm_spin = NoScrollSpinBox()
        vpm_spin.setRange(1, 100)
        try:
            vpm_spin.setValue(int(seed.get("vpm_rate", 1)))
        except (TypeError, ValueError):
            vpm_spin.setValue(1)
        vpm_row.addWidget(vpm_spin)
        vpm_row.addStretch()
        card_layout.addLayout(vpm_row)

        source_zones_list = self._create_checkbox_group_widget()
        card_layout.addWidget(QLabel("source_zones"))
        card_layout.addWidget(source_zones_list)

        origin_list = self._create_checkbox_group_widget()
        card_layout.addWidget(QLabel("origin"))
        card_layout.addWidget(origin_list)

        destination_list = self._create_checkbox_group_widget()
        card_layout.addWidget(QLabel("destination"))
        card_layout.addWidget(destination_list)

        card.setLayout(card_layout)
        self.weekday_schedule_cards_layout.addWidget(card)

        entry = {
            "widget": card,
            "name": name_input,
            "start_time": start_time,
            "end_time": end_time,
            "repeat_days": repeat_days_list,
            "vpm_rate": vpm_spin,
            "source_zones": source_zones_list,
            "origin": origin_list,
            "destination": destination_list,
        }
        self.weekday_schedule_entries.append(entry)
        self._apply_weekday_schedule_card_colors()

        selected_source = seed.get("source_zones", []) if isinstance(seed.get("source_zones", []), list) else []
        selected_origin = seed.get("origin", []) if isinstance(seed.get("origin", []), list) else []
        selected_destination = seed.get("destination", []) if isinstance(seed.get("destination", []), list) else []
        self._sync_multiselect_list(source_zones_list, self._get_zones_with_positive_allocation(), selected_source)
        self._sync_multiselect_list(origin_list, self._get_landmark_option_names(), selected_origin)
        self._sync_multiselect_list(destination_list, self._get_landmark_option_names(), selected_destination)

        remove_btn.clicked.connect(lambda: self.remove_weekday_schedule_card(card))
        name_input.textChanged.connect(lambda _: self._persist_current_config_safely())
        start_time.timeChanged.connect(lambda _: self._persist_current_config_safely())
        end_time.timeChanged.connect(lambda _: self._persist_current_config_safely())
        vpm_spin.valueChanged.connect(lambda _: self._persist_current_config_safely())
        self._persist_current_config_safely()

    def remove_weekday_schedule_card(self, card: QWidget):
        """Remove one weekday schedule subsection card."""
        self.weekday_schedule_entries = [e for e in self.weekday_schedule_entries if e["widget"] is not card]
        card.deleteLater()
        self._apply_weekday_schedule_card_colors()
        self._persist_current_config_safely()

    def load_weekday_schedule_from_config(self, weekday_schedule):
        """Load structured weekday schedule cards from config list."""
        self._clear_weekday_schedule_cards()
        if isinstance(weekday_schedule, list):
            for item in weekday_schedule:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    origin_values = item_copy.get("origin", [])
                    if isinstance(origin_values, list):
                        item_copy["origin"] = self._normalize_schedule_landmark_aliases_for_ui(origin_values)
                    destination_values = item_copy.get("destination", [])
                    if isinstance(destination_values, list):
                        item_copy["destination"] = self._normalize_schedule_landmark_aliases_for_ui(destination_values)
                    self.add_weekday_schedule_card(item_copy)
        self.refresh_weekday_schedule_options()

    def _collect_weekday_schedule_from_cards(self) -> List[Dict]:
        """Collect and validate structured weekday schedule sections."""
        output = []
        for idx, entry in enumerate(self.weekday_schedule_entries, start=1):
            name = entry["name"].text().strip()
            if not name:
                raise ValueError(f"weekday_schedule entry #{idx}: name is required.")

            start_qtime = entry["start_time"].time()
            end_qtime = entry["end_time"].time()
            start_str = start_qtime.toString("HH:mm")
            end_str = end_qtime.toString("HH:mm")
            if end_qtime <= start_qtime:
                raise ValueError(f"weekday_schedule entry '{name}': end_time must be later than start_time.")

            repeat_days = sorted([
                int(value)
                for value in self._get_selected_zone_names_from_list(entry["repeat_days"])
                if str(value).isdigit()
            ])
            if not repeat_days:
                raise ValueError(f"weekday_schedule entry '{name}': select at least one day in repeat_on_days.")

            source_zones = self._get_selected_zone_names_from_list(entry["source_zones"])
            if not source_zones:
                raise ValueError(f"weekday_schedule entry '{name}': select at least one source_zones value.")

            origin = self._get_selected_zone_names_from_list(entry["origin"])
            if not origin:
                raise ValueError(f"weekday_schedule entry '{name}': select at least one origin landmark.")
            origin = self._expand_schedule_landmark_aliases(origin)

            destination = self._get_selected_zone_names_from_list(entry["destination"])
            if not destination:
                raise ValueError(f"weekday_schedule entry '{name}': select at least one destination landmark.")
            destination = self._expand_schedule_landmark_aliases(destination)

            output.append({
                "name": name,
                "start_time": start_str,
                "end_time": end_str,
                "repeat_on_days": repeat_days,
                "vpm_rate": int(entry["vpm_rate"].value()),
                "source_zones": source_zones,
                "origin": origin,
                "destination": destination,
            })
        return output

    def _clear_detected_zone_rows(self):
        """Remove detected-zone row widgets from the UI."""
        while self.detected_zones_list_layout.count():
            item = self.detected_zones_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.detected_zone_row_widgets = {}

    def detect_zones_from_network(self):
        """Scan loaded network data and detect zones from edge/node IDs."""
        if not self.network_parser:
            QMessageBox.warning(self, "No Network", "Load a network first, then detect zones.")
            return

        edges = self.network_parser.get_edges()
        # Rendered nodes are based on get_nodes() if present, otherwise junctions.
        node_source = self.network_parser.get_nodes()
        if not node_source:
            node_source = self.network_parser.get_junctions()

        edge_zone_map, node_zone_map = self._load_zone_attributes_from_network_file()
        has_explicit_zone_attr = bool(edge_zone_map or node_zone_map)

        zone_edges = {}
        zone_nodes = {}

        for edge_id, edge_data in edges.items():
            zone_name = edge_zone_map.get(edge_id)
            if not zone_name:
                zone_name = node_zone_map.get(edge_data.get('from', ''))
            if not zone_name:
                zone_name = node_zone_map.get(edge_data.get('to', ''))
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_id)
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_data.get('from', ''))
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(edge_data.get('to', ''))
            if not zone_name:
                continue
            zone_edges.setdefault(zone_name, set()).add(edge_id)

        for node_id in node_source.keys():
            zone_name = node_zone_map.get(node_id)
            if not zone_name:
                zone_name = self._extract_zone_name_from_id(node_id)
            if not zone_name:
                continue
            zone_nodes.setdefault(zone_name, set()).add(node_id)

        all_zone_names = sorted(set(zone_edges.keys()) | set(zone_nodes.keys()))
        self.detected_zones = {}
        self.detected_visible_zones = set()
        self._clear_detected_zone_rows()

        palette = [
            QColor(220, 20, 60),   # crimson
            QColor(30, 144, 255),  # dodger blue
            QColor(50, 205, 50),   # lime green
            QColor(255, 140, 0),   # dark orange
            QColor(148, 0, 211),   # dark violet
            QColor(255, 99, 71),   # tomato
            QColor(0, 206, 209),   # dark turquoise
            QColor(255, 215, 0),   # gold
        ]

        for idx, zone_name in enumerate(all_zone_names):
            color = palette[idx % len(palette)]
            edge_ids = zone_edges.get(zone_name, set())
            node_ids = zone_nodes.get(zone_name, set())
            self.detected_zones[zone_name] = {
                'edges': edge_ids,
                'nodes': node_ids,
                'color': color,
            }
            row_widget = self._create_detected_zone_row(zone_name, len(edge_ids), len(node_ids))
            self.detected_zones_list_layout.addWidget(row_widget)

        if not all_zone_names:
            self.detected_zones_status.setText("No zones detected from edge/node IDs.")
        else:
            mode_label = "attribute-based" if has_explicit_zone_attr else "ID-heuristic"
            self.detected_zones_status.setText(f"Detected {len(all_zone_names)} zone(s) ({mode_label}).")

        self.refresh_zone_allocation_section()
        self.refresh_weekday_schedule_options()
        self._apply_color_overlays()
        self._persist_current_config_safely()

    def detect_landmarks_from_network(self):
        """Detect landmarks from net.xml edge landmark params."""
        if not self.network_parser:
            return
        landmarks_map = self._load_landmarks_from_network_file()
        self._set_landmarks_from_mapping(landmarks_map)
        self.refresh_default_landmark_zone_options()
        self.refresh_weekday_schedule_options()
        self._apply_color_overlays()
        self._persist_current_config_safely()

    def _create_detected_zone_row(self, zone_name: str, edge_count: int, node_count: int) -> QWidget:
        """Create one detected-zone line: zone - edges - nodes - show/hide."""
        row = QFrame()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)

        color = self.detected_zones[zone_name]['color']
        color_chip = QLabel()
        color_chip.setFixedSize(12, 12)
        color_chip.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: 1px solid #444; border-radius: 2px;"
        )
        row_layout.addWidget(color_chip)

        row_layout.addWidget(QLabel(zone_name))
        row_layout.addWidget(QLabel(f"- {edge_count} edges"))
        row_layout.addWidget(QLabel(f"- {node_count} nodes"))
        row_layout.addStretch()

        toggle_btn = QPushButton("Show")
        toggle_btn.setCheckable(True)
        toggle_btn.toggled.connect(lambda checked, z=zone_name, b=toggle_btn: self.toggle_detected_zone(z, checked, b))
        row_layout.addWidget(toggle_btn)

        row.setLayout(row_layout)
        self.detected_zone_row_widgets[zone_name] = row
        return row

    def _create_detected_landmark_row(self, landmark_name: str, edge_count: int) -> QWidget:
        """Create one detected-landmark line: name - edges - show/hide."""
        row = QFrame()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)

        color = self.detected_landmarks[landmark_name]['color']
        color_chip = QLabel()
        color_chip.setFixedSize(12, 12)
        color_chip.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: 1px solid #444; border-radius: 2px;"
        )
        row_layout.addWidget(color_chip)
        row_layout.addWidget(QLabel(landmark_name))
        row_layout.addWidget(QLabel(f"- {edge_count} edges"))
        row_layout.addStretch()

        toggle_btn = QPushButton("Show")
        toggle_btn.setCheckable(True)
        toggle_btn.toggled.connect(
            lambda checked, l=landmark_name, b=toggle_btn: self.toggle_detected_landmark(l, checked, b)
        )
        row_layout.addWidget(toggle_btn)

        row.setLayout(row_layout)
        self.detected_landmark_row_widgets[landmark_name] = row
        return row

    def toggle_detected_zone(self, zone_name: str, checked: bool, button: QPushButton):
        """Show/hide one detected zone on the map with its unique color."""
        if zone_name not in self.detected_zones:
            return

        if checked:
            self.detected_visible_zones.add(zone_name)
            button.setText("Hide")
        else:
            self.detected_visible_zones.discard(zone_name)
            button.setText("Show")

        self._apply_color_overlays()

    def toggle_detected_landmark(self, landmark_name: str, checked: bool, button: QPushButton):
        """Show/hide one detected landmark edges with unique color."""
        if landmark_name not in self.detected_landmarks:
            return
        if checked:
            self.detected_visible_landmarks.add(landmark_name)
            button.setText("Hide")
        else:
            self.detected_visible_landmarks.discard(landmark_name)
            button.setText("Show")
        self._apply_color_overlays()

    def _apply_color_overlays(self):
        """Apply visible zone and landmark color overlays to map entities."""
        edge_colors = {}
        node_colors = {}
        # Zones: edges + nodes
        for zone_name in self.detected_visible_zones:
            zone_data = self.detected_zones.get(zone_name)
            if not zone_data:
                continue
            color = zone_data['color']
            for edge_id in zone_data['edges']:
                edge_colors[edge_id] = color
            for node_id in zone_data['nodes']:
                node_colors[node_id] = color
        # Landmarks: edges only (override zone edge colors when both active).
        for landmark_name in self.detected_visible_landmarks:
            landmark_data = self.detected_landmarks.get(landmark_name)
            if not landmark_data:
                continue
            color = landmark_data['color']
            for edge_id in landmark_data['edges']:
                edge_colors[edge_id] = color

        self.map_view.set_edge_color_overrides(edge_colors)
        self.map_view.set_node_color_overrides(node_colors)
        self.map_view.viewport().update()

    def add_vehicle_type_card(
        self,
        name: str = "",
        length: float = 4.5,
        width: float = 1.8,
        height: float = 1.5,
        max_speed: float = 30.0,
        color: str = "blue",
    ):
        """Add one vehicle type card with editable fields."""
        card_container = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        type_group = QGroupBox()
        type_form = QFormLayout()
        type_form.setContentsMargins(8, 8, 8, 8)
        type_form.setSpacing(6)

        name_input = QLineEdit(name)
        length_spin = NoScrollDoubleSpinBox()
        length_spin.setRange(0.1, 100.0)
        length_spin.setDecimals(2)
        length_spin.setValue(length)

        width_spin = NoScrollDoubleSpinBox()
        width_spin.setRange(0.1, 20.0)
        width_spin.setDecimals(2)
        width_spin.setValue(width)

        height_spin = NoScrollDoubleSpinBox()
        height_spin.setRange(0.1, 20.0)
        height_spin.setDecimals(2)
        height_spin.setValue(height)

        speed_spin = NoScrollDoubleSpinBox()
        speed_spin.setRange(0.1, 100.0)
        speed_spin.setDecimals(2)
        speed_spin.setValue(max_speed)

        color_input = QLineEdit(color)

        type_form.addRow("name", name_input)
        type_form.addRow("length", length_spin)
        type_form.addRow("width", width_spin)
        type_form.addRow("height", height_spin)
        type_form.addRow("max_speed", speed_spin)
        type_form.addRow("color", color_input)
        type_group.setLayout(type_form)

        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("Remove vehicle type")
        remove_btn.clicked.connect(lambda: self.remove_vehicle_type_card(card_container))

        row_layout.addWidget(type_group, stretch=1)
        row_layout.addWidget(remove_btn)
        card_container.setLayout(row_layout)

        entry = {
            "container": card_container,
            "group": type_group,
            "name": name_input,
            "length": length_spin,
            "width": width_spin,
            "height": height_spin,
            "max_speed": speed_spin,
            "color": color_input,
        }
        self.vehicle_type_entries.append(entry)
        self.vehicle_types_cards_layout.addWidget(card_container)
        self._apply_vehicle_type_card_colors()
        self.refresh_zone_allocation_section()
        self._persist_current_config_safely()

    def remove_vehicle_type_card(self, card_container: QWidget):
        """Remove one vehicle type card."""
        for idx, entry in enumerate(self.vehicle_type_entries):
            if entry["container"] is card_container:
                self.vehicle_types_cards_layout.removeWidget(card_container)
                card_container.deleteLater()
                self.vehicle_type_entries.pop(idx)
                self._apply_vehicle_type_card_colors()
                self.refresh_zone_allocation_section()
                self._persist_current_config_safely()
                break

    def _apply_vehicle_type_card_colors(self):
        """Apply alternating background colors to vehicle type cards."""
        palette = [
            "#e8f1ff",  # blue
            "#e9fbe7",  # green
            "#fff4dc",  # peach
            "#efe8ff",  # purple
            "#e6fbf7",  # mint
            "#ffe8ef",  # pink
        ]
        for idx, entry in enumerate(self.vehicle_type_entries):
            group = entry.get("group")
            if group is None:
                continue
            color = palette[idx % len(palette)]
            group.setAttribute(Qt.WA_StyledBackground, True)
            group.setStyleSheet(
                f"QGroupBox {{ background-color: {color}; border: 1px solid #d5d5d5; border-radius: 6px; }}"
            )

    def _collect_vehicle_types_from_cards(self) -> Dict[str, Dict]:
        """Collect vehicle types from card widgets."""
        vehicle_types: Dict[str, Dict] = {}
        for entry in self.vehicle_type_entries:
            name = entry["name"].text().strip()
            if not name:
                raise ValueError("Vehicle type name cannot be empty.")
            if name in vehicle_types:
                raise ValueError(f"Duplicate vehicle type name: {name}")
            vehicle_types[name] = {
                "length": float(entry["length"].value()),
                "width": float(entry["width"].value()),
                "height": float(entry["height"].value()),
                "max_speed": float(entry["max_speed"].value()),
                "color": entry["color"].text().strip() or "gray",
            }

        if not vehicle_types:
            raise ValueError("At least one vehicle type is required.")
        return vehicle_types

    def _get_vehicle_type_names(self) -> List[str]:
        """Return current vehicle type names from cards."""
        names = []
        for entry in self.vehicle_type_entries:
            name = entry["name"].text().strip()
            if name:
                names.append(name)
        return names

    def _clear_zone_allocation_cards(self):
        """Clear all zone allocation card widgets."""
        while self.zone_allocation_cards_layout.count():
            item = self.zone_allocation_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.zone_allocation_entries = {}

    def _get_existing_zone_allocation_data(self) -> Dict[str, Dict]:
        """Capture existing zone allocation values before rebuilding UI."""
        existing = {}
        for zone_name, entry in self.zone_allocation_entries.items():
            dist_values = {
                vt: spin.value()
                for vt, spin in entry.get("distribution_spins", {}).items()
            }
            existing[zone_name] = {
                "percentage": entry["percentage_spin"].value(),
                "distribution": dist_values,
                "has_distribution": entry.get("has_distribution", True),
            }
        return existing

    def refresh_zone_allocation_section(self, preserve_existing: bool = True):
        """Rebuild zone allocation cards using detected zones and vehicle types."""
        if not hasattr(self, "zone_allocation_cards_layout"):
            return

        existing = self._get_existing_zone_allocation_data() if preserve_existing else {}
        seed = self._zone_allocation_seed if isinstance(self._zone_allocation_seed, dict) else {}
        vehicle_type_names = self._get_vehicle_type_names()

        zone_names = set(self.detected_zones.keys()) if self.detected_zones else set()
        zone_names.update(seed.keys())
        # Keep a noise bucket aligned with desired JSON structure.
        zone_names.add("noise")
        zone_names = sorted(zone_names)

        self._clear_zone_allocation_cards()
        for zone_name in zone_names:
            prior = existing.get(zone_name, {})
            seed_zone = seed.get(zone_name, {}) if isinstance(seed.get(zone_name, {}), dict) else {}
            prior_percentage = prior.get("percentage", seed_zone.get("percentage", 0.0))
            prior_distribution = prior.get(
                "distribution",
                seed_zone.get("vehicle_type_distribution", {}) if isinstance(seed_zone.get("vehicle_type_distribution", {}), dict) else {}
            )
            has_distribution = zone_name.lower() != "noise"
            if "has_distribution" in prior:
                has_distribution = bool(prior["has_distribution"])
            self._add_zone_allocation_card(
                zone_name=zone_name,
                vehicle_type_names=vehicle_type_names,
                percentage=float(prior_percentage),
                distribution=prior_distribution,
                has_distribution=has_distribution,
            )
        self._update_zone_allocation_total_label()
        self.refresh_default_landmark_zone_options()
        self.refresh_weekday_schedule_options()

    def _add_zone_allocation_card(
        self,
        zone_name: str,
        vehicle_type_names: List[str],
        percentage: float = 0.0,
        distribution: Optional[Dict[str, float]] = None,
        has_distribution: bool = True,
    ):
        """Create one zone allocation card for JSON structure."""
        distribution = distribution or {}
        card = QGroupBox(zone_name)
        card_layout = QVBoxLayout()
        card_layout.setSpacing(6)

        percentage_row = QHBoxLayout()
        percentage_row.addWidget(QLabel("percentage"))
        percentage_spin = NoScrollDoubleSpinBox()
        percentage_spin.setRange(0.0, 100.0)
        percentage_spin.setDecimals(2)
        percentage_spin.setValue(float(percentage))
        percentage_row.addWidget(percentage_spin)
        percentage_row.addStretch()
        card_layout.addLayout(percentage_row)

        distribution_spins = {}
        if has_distribution:
            card_layout.addWidget(QLabel("vehicle_type_distribution"))
            for vt_name in vehicle_type_names:
                row = QHBoxLayout()
                row.addWidget(QLabel(vt_name))
                spin = NoScrollDoubleSpinBox()
                spin.setRange(0.0, 100.0)
                spin.setDecimals(2)
                if distribution:
                    spin.setValue(float(distribution.get(vt_name, 0.0)))
                elif vt_name == "passenger":
                    spin.setValue(100.0)
                else:
                    spin.setValue(0.0)
                spin.valueChanged.connect(lambda _v, z=zone_name, vt=vt_name: self._on_distribution_spin_changed(z, vt))
                row.addWidget(spin)
                row.addStretch()
                card_layout.addLayout(row)
                distribution_spins[vt_name] = spin

            distribution_total_label = QLabel("Distribution total: 0%")
            distribution_total_label.setStyleSheet("color: #666; font-size: 10px;")
            card_layout.addWidget(distribution_total_label)
        else:
            distribution_total_label = None

        card.setLayout(card_layout)
        self.zone_allocation_cards_layout.addWidget(card)
        self.zone_allocation_entries[zone_name] = {
            "widget": card,
            "percentage_spin": percentage_spin,
            "distribution_spins": distribution_spins,
            "has_distribution": has_distribution,
            "distribution_total_label": distribution_total_label,
        }
        percentage_spin.valueChanged.connect(lambda _v, z=zone_name: self._on_zone_percentage_changed(z))
        if has_distribution:
            self._update_distribution_total_label(zone_name)

    def _on_zone_percentage_changed(self, changed_zone: str):
        """Handle zone percentage updates."""
        if self._suppress_auto_persist:
            return
        self._update_zone_allocation_total_label()
        self._persist_current_config_safely()

    def _update_zone_allocation_total_label(self):
        """Update total percentage label for all zones."""
        total = 0.0
        for entry in self.zone_allocation_entries.values():
            total += entry["percentage_spin"].value()
        self.zone_allocation_total_label.setText(f"Total: {total:.2f}%")
        if abs(total - 100.0) <= 0.01:
            self.zone_allocation_total_label.setStyleSheet("color: #2e7d32; font-size: 10px;")
        else:
            self.zone_allocation_total_label.setStyleSheet("color: #c62828; font-size: 10px;")

    def _update_distribution_total_label(self, zone_name: str):
        """Update distribution total label for one zone."""
        entry = self.zone_allocation_entries.get(zone_name)
        if not entry or not entry.get("has_distribution", True):
            return
        total = sum(spin.value() for spin in entry["distribution_spins"].values())
        label = entry.get("distribution_total_label")
        if label is None:
            return
        label.setText(f"Distribution total: {total:.2f}%")
        if abs(total - 100.0) <= 0.01:
            label.setStyleSheet("color: #2e7d32; font-size: 10px;")
        else:
            label.setStyleSheet("color: #c62828; font-size: 10px;")

    def _on_distribution_spin_changed(self, zone_name: str, changed_type: str):
        """Handle distribution updates and refresh total label."""
        if self._suppress_auto_persist:
            return
        self._update_distribution_total_label(zone_name)
        self._persist_current_config_safely()

    def _collect_zone_allocation_from_cards(self) -> Dict[str, Dict]:
        """Collect zone_allocation dict in target JSON structure."""
        zone_allocation = {}
        total_percentage = 0.0
        for zone_name, entry in self.zone_allocation_entries.items():
            zone_payload = {
                "percentage": float(entry["percentage_spin"].value())
            }
            total_percentage += zone_payload["percentage"]
            if entry.get("has_distribution", True):
                distribution = {
                    vt_name: float(spin.value())
                    for vt_name, spin in entry["distribution_spins"].items()
                }
                if abs(sum(distribution.values()) - 100.0) > 0.01:
                    raise ValueError(f"vehicle_type_distribution for zone '{zone_name}' must total 100%.")
                zone_payload["vehicle_type_distribution"] = distribution
            zone_allocation[zone_name] = zone_payload
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError("Sum of all zone percentages (including noise) must be 100%.")
        return zone_allocation

    def _build_config_from_form(self) -> dict:
        """Build simulation config dict from GUI inputs."""
        weekday_schedule = self._collect_weekday_schedule_from_cards()
        landmarks = self._collect_landmarks_for_config()
        zone_allocation = self._collect_zone_allocation_from_cards()
        vehicle_types = self._collect_vehicle_types_from_cards()

        config = {
            "snapshot_dir": self.snapshot_dir_input.text().strip(),
            "snapshot_interval_sec": int(self.snapshot_interval_spin.value()),
            "vehicle_generation": {
                "simulation_weeks": int(self.simulation_weeks_spin.value()),
                "total_num_vehicles": int(self.total_vehicles_spin.value()),
                "dev_fraction": float(self.dev_fraction_spin.value()),
                "zone_allocation": zone_allocation,
                "vehicle_types": vehicle_types
            },
            "landmarks": landmarks,
            "weekday_schedule": weekday_schedule
        }
        return config

    def _collect_landmarks_for_config(self) -> Dict[str, List[str]]:
        """Collect landmarks as JSON-serializable mapping used by simulation config."""
        source = self.detected_landmarks
        if not source and isinstance(self._landmarks_seed, dict):
            source = {
                name: {"edges": set(edges) if isinstance(edges, list) else set(edges or []), "color": QColor(120, 120, 120)}
                for name, edges in self._landmarks_seed.items()
            }
        output = {}
        for landmark_name in sorted(source.keys()):
            edge_ids = source[landmark_name].get("edges", set())
            output[landmark_name] = sorted(list(edge_ids))
        output["default_landmarks"] = self._collect_default_landmarks_from_form()
        return output

    def _persist_current_config_safely(self):
        """Try to persist current form into project config; ignore transient invalid edits."""
        if self._suppress_auto_persist:
            return
        try:
            config = self._build_config_from_form()
            self._write_project_simulation_config(config)
        except Exception:
            # Ignore while user is still editing incomplete data.
            pass

    def on_map_network_render_finished(self):
        """Restore current selection/display visualization after network redraw."""
        if self.zones_locked:
            self.map_view.set_selected_edges(set())
            self.map_view.set_selected_nodes(set())
        elif self.displayed_zones:
            self._update_display_visualization()
        else:
            self._update_zone_visualization()

        # Re-apply zone/landmark coloring overlays after map items are rebuilt.
        self._apply_color_overlays()
        self.map_view.viewport().update()
    
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
        delete_btn = QPushButton("✗")
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
        self.save_route_generation_settings()
    
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
    
    def lock_zones(self, show_dialogs: bool = True, offer_unselected_assignment: bool = True):
        """Lock zones: disable editing, remove colored areas, reset highlighting."""
        if self.zones_locked:
            return
        
        if not self.network_parser:
            if show_dialogs:
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
        if offer_unselected_assignment and (unselected_edges or unselected_junctions):
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
        self.save_route_generation_settings()
        
        # Update map view to show locked state (zone names in centers)
        self.map_view.viewport().update()
        
        # Update statistics
        self.update_statistics()
        
        if show_dialogs:
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
        self.save_route_generation_settings()
        
        # Update map view to show unlocked state (zone names at corners)
        self.map_view.viewport().update()

    def _restore_route_generation_state(self):
        """Restore persisted route generation UI state after zones are loaded."""
        # Keep only valid zone IDs (zones may have been deleted from config).
        self.displayed_zones = {zone_id for zone_id in self.displayed_zones if zone_id in self.zones}

        if self._restore_zones_locked:
            self.lock_zones(show_dialogs=False, offer_unselected_assignment=False)
        elif self.displayed_zones:
            self._update_display_visualization()
            self.map_view.viewport().update()
        else:
            self._update_zone_visualization()
            self.map_view.viewport().update()
    
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
        self.save_route_generation_settings()

    def save_route_generation_settings(self):
        """Persist route generation page UI selections to project config."""
        settings = {
            'displayed_zones': sorted([zone_id for zone_id in self.displayed_zones if zone_id in self.zones]),
            'zones_locked': bool(self.zones_locked),
        }
        self.config_manager.save_route_generation_settings(settings)
