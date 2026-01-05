"""
Debug page for visualizing the first trajectory with trimming applied.
Simple page with just map, network, GPS points, and connecting line.
"""

import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (QDoubleSpinBox, QGraphicsEllipseItem,
                               QGraphicsPolygonItem, QGraphicsTextItem,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.project_manager import _get_project_root
from src.utils.trip_validator import (TripValidationResult,
                                      validate_trip_segments)


class DebugTrajectoryPage(QWidget):
    """Debug page for visualizing first trajectory with trimming."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.network_parser = None
        self.sumo_net = None
        self._edge_spatial_index = None
        self._route_items = []
        self._segment_items = []  # Store segment items separately for clearing
        self.train_csv_path = None  # To store the resolved train.csv path
        self._bounding_box_polygon = None  # Store bounding box polygon for edge intersection checks
        self._bounding_box_params = None  # Store bounding box parameters (center_x, center_y, width, height, angle)
        self._current_sumo_points = None  # Store current trajectory points (Y-flipped for display)
        
        self.init_ui()
        
        # Auto-load network (trajectory will be loaded via Show button)
        QTimer.singleShot(100, self.load_network_and_trajectory)
    
    def init_ui(self):
        """Initialize the UI - just map and network."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header with back button and title
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        back_btn = QPushButton("← Back to Home")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        header_layout.addWidget(back_btn)
        
        header_layout.addStretch()
        
        # Title
        title = QLabel(f"🔧 Debug: First Trajectory - {self.project_name}")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #FF9800; padding: 10px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Map view (takes all available space)
        self.map_view = SimulationView()
        self.map_view.setMinimumSize(800, 600)
        main_layout.addWidget(self.map_view)
        
        # Footer with coordinate conversion adjustment controls
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(10, 5, 10, 5)
        footer_layout.setSpacing(10)
        
        # Coordinate conversion adjustment controls
        conv_adjust_label = QLabel("Conv adjust:")
        conv_adjust_label.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        footer_layout.addWidget(conv_adjust_label)
        
        # X adjustment
        x_adj_label = QLabel("X:")
        x_adj_label.setStyleSheet("color: #333; font-size: 10px;")
        footer_layout.addWidget(x_adj_label)
        
        self.conv_adjust_x_spinbox = QDoubleSpinBox()
        self.conv_adjust_x_spinbox.setRange(-0.1, 0.1)
        self.conv_adjust_x_spinbox.setSingleStep(0.001)
        self.conv_adjust_x_spinbox.setValue(0.0)
        self.conv_adjust_x_spinbox.setDecimals(4)
        self.conv_adjust_x_spinbox.setToolTip("Adjust X coordinate conversion (normalized offset)")
        self.conv_adjust_x_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 70px;
            }
        """)
        self.conv_adjust_x_spinbox.valueChanged.connect(self.on_conv_adjust_changed)
        footer_layout.addWidget(self.conv_adjust_x_spinbox)
        
        # Y adjustment
        y_adj_label = QLabel("Y:")
        y_adj_label.setStyleSheet("color: #333; font-size: 10px;")
        footer_layout.addWidget(y_adj_label)
        
        self.conv_adjust_y_spinbox = QDoubleSpinBox()
        self.conv_adjust_y_spinbox.setRange(-0.1, 0.1)
        self.conv_adjust_y_spinbox.setSingleStep(0.001)
        self.conv_adjust_y_spinbox.setValue(0.0)
        self.conv_adjust_y_spinbox.setDecimals(4)
        self.conv_adjust_y_spinbox.setToolTip("Adjust Y coordinate conversion (normalized offset)")
        self.conv_adjust_y_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 70px;
            }
        """)
        self.conv_adjust_y_spinbox.valueChanged.connect(self.on_conv_adjust_changed)
        footer_layout.addWidget(self.conv_adjust_y_spinbox)
        
        footer_layout.addSpacing(15)
        
        # Scale factors
        conv_scale_label = QLabel("Scale:")
        conv_scale_label.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        footer_layout.addWidget(conv_scale_label)
        
        # X scale
        x_scale_label = QLabel("X:")
        x_scale_label.setStyleSheet("color: #333; font-size: 10px;")
        footer_layout.addWidget(x_scale_label)
        
        self.conv_scale_x_spinbox = QDoubleSpinBox()
        self.conv_scale_x_spinbox.setRange(0.9, 1.1)
        self.conv_scale_x_spinbox.setSingleStep(0.001)
        self.conv_scale_x_spinbox.setValue(1.0)
        self.conv_scale_x_spinbox.setDecimals(4)
        self.conv_scale_x_spinbox.setToolTip("Scale X coordinate conversion")
        self.conv_scale_x_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 70px;
            }
        """)
        self.conv_scale_x_spinbox.valueChanged.connect(self.on_conv_adjust_changed)
        footer_layout.addWidget(self.conv_scale_x_spinbox)
        
        # Y scale
        y_scale_label = QLabel("Y:")
        y_scale_label.setStyleSheet("color: #333; font-size: 10px;")
        footer_layout.addWidget(y_scale_label)
        
        self.conv_scale_y_spinbox = QDoubleSpinBox()
        self.conv_scale_y_spinbox.setRange(0.9, 1.1)
        self.conv_scale_y_spinbox.setSingleStep(0.001)
        self.conv_scale_y_spinbox.setValue(1.0)
        self.conv_scale_y_spinbox.setDecimals(4)
        self.conv_scale_y_spinbox.setToolTip("Scale Y coordinate conversion")
        self.conv_scale_y_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 70px;
            }
        """)
        self.conv_scale_y_spinbox.valueChanged.connect(self.on_conv_adjust_changed)
        footer_layout.addWidget(self.conv_scale_y_spinbox)
        
        footer_layout.addSpacing(15)
        
        # Trajectory selection controls
        traj_label = QLabel("Trajectory:")
        traj_label.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        footer_layout.addWidget(traj_label)
        
        self.trajectory_spinbox = QDoubleSpinBox()
        self.trajectory_spinbox.setRange(1, 1)  # Will be updated when CSV is loaded
        self.trajectory_spinbox.setSingleStep(1)
        self.trajectory_spinbox.setValue(1)
        self.trajectory_spinbox.setDecimals(0)
        self.trajectory_spinbox.setToolTip("Select trajectory number to display")
        self.trajectory_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 11px;
                padding: 3px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 80px;
            }
        """)
        footer_layout.addWidget(self.trajectory_spinbox)
        
        # Show button
        self.show_btn = QPushButton("Show")
        self.show_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.show_btn.clicked.connect(self.on_show_clicked)
        self.show_btn.setEnabled(False)  # Disabled until network is loaded
        footer_layout.addWidget(self.show_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_btn.clicked.connect(self.on_clear_clicked)
        footer_layout.addWidget(self.clear_btn)
        
        footer_layout.addSpacing(15)
        
        # Segment selection controls
        segment_label = QLabel("Segment:")
        segment_label.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        footer_layout.addWidget(segment_label)
        
        self.segment_spinbox = QDoubleSpinBox()
        self.segment_spinbox.setRange(1, 1)  # Will be updated when trajectory is loaded
        self.segment_spinbox.setSingleStep(1)
        self.segment_spinbox.setValue(1)
        self.segment_spinbox.setDecimals(0)
        self.segment_spinbox.setToolTip("Select segment number (1 = GPS points 1-2, 2 = GPS points 2-3, etc.)")
        self.segment_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 11px;
                padding: 3px;
                border: 1px solid #999;
                border-radius: 3px;
                background-color: white;
                min-width: 80px;
            }
        """)
        footer_layout.addWidget(self.segment_spinbox)
        
        # Show Segment button
        self.show_segment_btn = QPushButton("Show Segment")
        self.show_segment_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.show_segment_btn.clicked.connect(self.on_show_segment_clicked)
        self.show_segment_btn.setEnabled(False)  # Disabled until trajectory is loaded
        footer_layout.addWidget(self.show_segment_btn)
        
        # Clear Segment button
        self.clear_segment_btn = QPushButton("Clear Segment")
        self.clear_segment_btn.setStyleSheet("""
            QPushButton {
                background-color: #E91E63;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #C2185B;
            }
        """)
        self.clear_segment_btn.clicked.connect(self.on_clear_segment_clicked)
        footer_layout.addWidget(self.clear_segment_btn)
        
        footer_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Loading network...")
        self.status_label.setStyleSheet("color: #333; font-size: 12px; padding: 5px;")
        footer_layout.addWidget(self.status_label)
        
        main_layout.addLayout(footer_layout)
        
        self.setLayout(main_layout)
    
    def log(self, message: str):
        """Simple log to status label and console."""
        print(f"[DEBUG] {message}")  # Also print to console for debugging
        self.status_label.setText(message)
    
    def on_conv_adjust_changed(self, value: float):
        """Handle coordinate conversion adjustment change."""
        if self.network_parser:
            adjust_x = self.conv_adjust_x_spinbox.value()
            adjust_y = self.conv_adjust_y_spinbox.value()
            scale_x = self.conv_scale_x_spinbox.value()
            scale_y = self.conv_scale_y_spinbox.value()
            
            self.network_parser.conv_adjust_x = adjust_x
            self.network_parser.conv_adjust_y = adjust_y
            self.network_parser.conv_scale_x = scale_x
            self.network_parser.conv_scale_y = scale_y
            
            # Reload OSM tiles to apply new conversion
            if self.map_view and self.map_view.show_osm_map:
                self.map_view._load_osm_tiles()
            
            self.log(f"Conv adjust: X={adjust_x:.4f}, Y={adjust_y:.4f}, Scale X={scale_x:.4f}, Y={scale_y:.4f}")
    
    def load_network_and_trajectory(self):
        """Load network and first trajectory using paths from project_info.json and settings.json."""
        self.log("Loading network...")
        
        import json
        project_path = Path(self.project_path)
        
        network_file = None
        train_csv = None
        
        # First, try loading from project_info.json
        project_info_file = project_path / 'project_info.json'
        if project_info_file.exists():
            try:
                with open(project_info_file, 'r', encoding='utf-8') as f:
                    project_info = json.load(f)
                
                # Get network file path from project info
                if 'network_file' in project_info:
                    network_path = Path(project_info['network_file'])
                    if network_path.is_absolute():
                        network_file = network_path
                    else:
                        network_file = project_path / network_path
                elif 'network_path' in project_info:
                    network_path = Path(project_info['network_path'])
                    if network_path.is_absolute():
                        network_file = network_path
                    else:
                        network_file = project_path / network_path
                
                # Get train.csv path from project info
                if 'train_csv' in project_info:
                    train_path = Path(project_info['train_csv'])
                    if train_path.is_absolute():
                        train_csv = train_path
                    else:
                        train_csv = project_path / train_path
                elif 'train_path' in project_info:
                    train_path = Path(project_info['train_path'])
                    if train_path.is_absolute():
                        train_csv = train_path
                    else:
                        train_csv = project_path / train_path
                
                self.log(f"Loaded project info from {project_info_file.name}")
            except Exception as e:
                self.log(f"⚠️ Error reading project_info.json: {e}")
        
        # Also check settings.json (like dataset_conversion_page does)
        config_dir = project_path / 'config'
        settings_file = config_dir / 'settings.json'
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Get train_csv_path from settings
                if 'train_csv_path' in settings and settings['train_csv_path']:
                    train_path = Path(settings['train_csv_path'])
                    if train_path.exists():
                        train_csv = train_path
                        self.log(f"Found train.csv in settings.json: {train_csv}")
            except Exception as e:
                self.log(f"⚠️ Error reading settings.json: {e}")
        
        # Fallback: try common locations if not found in JSON files
        if not network_file:
            # Try project_path/config/porto.net.xml
            config_net = project_path / 'config' / 'porto.net.xml'
            if config_net.exists():
                network_file = config_net
            else:
                # Try searching for any .net.xml file
                network_files = list(project_path.glob("**/*.net.xml"))
                if network_files:
                    network_file = network_files[0]
        
        if not network_file or not network_file.exists():
            self.log("❌ No network file found (.net.xml)")
            self.log("  Checked: project_info.json, settings.json, config/porto.net.xml, and **/*.net.xml")
            return
        
        self.log(f"Found network: {network_file}")
        
        # Load network parser
        try:
            self.network_parser = NetworkParser(str(network_file))
            self.map_view.load_network(self.network_parser, roads_junctions_only=False)
            
            # Initialize coordinate conversion adjustments from current spinbox values
            if self.network_parser:
                self.network_parser.conv_adjust_x = self.conv_adjust_x_spinbox.value()
                self.network_parser.conv_adjust_y = self.conv_adjust_y_spinbox.value()
                self.network_parser.conv_scale_x = self.conv_scale_x_spinbox.value()
                self.network_parser.conv_scale_y = self.conv_scale_y_spinbox.value()
            
            # Load SUMO network for route calculation
            try:
                import os

                import sumolib
                if 'SUMO_HOME' not in os.environ:
                    self.log("⚠️ SUMO_HOME not set, route calculation will be limited")
                else:
                    self.sumo_net = sumolib.net.readNet(str(network_file))
                    self.log(f"✓ SUMO network loaded for routing: {len(self.sumo_net.getEdges())} edges")
            except ImportError:
                self.log("⚠️ sumolib not available. Route calculation will not work.")
                self.sumo_net = None
            except Exception as e:
                self.log(f"⚠️ Failed to load SUMO network for routing: {e}")
                self.sumo_net = None
            
            self.log("✓ Network loaded")
        except Exception as e:
            self.log(f"❌ Failed to load network: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Enable OSM map
        self.map_view.set_osm_map_visible(True)
        
        # Fallback for train_csv if not found in JSON files
        if not train_csv or not train_csv.exists():
            # Try project_path/train.csv
            train_csv = project_path / "train.csv"
            
            # Try project_path/dataset/train.csv
            if not train_csv.exists():
                train_csv = project_path / "dataset" / "train.csv"
            
            # Try Porto/dataset/train.csv (use workspace root)
            if not train_csv.exists():
                workspace_root = _get_project_root()
                porto_train = workspace_root / 'Porto' / 'dataset' / 'train.csv'
                if porto_train.exists():
                    train_csv = porto_train
                    self.log(f"Found train.csv in Porto/dataset: {train_csv}")
                else:
                    # Also try parent.parent in case project is in a subdirectory
                    porto_train = project_path.parent.parent / 'Porto' / 'dataset' / 'train.csv'
                    if porto_train.exists():
                        train_csv = porto_train
                        self.log(f"Found train.csv in Porto/dataset (via parent.parent): {train_csv}")
        
        # Store train_csv path for later use
        self.train_csv_path = train_csv
        
        # Count trajectories and update UI
        if train_csv and train_csv.exists():
            trajectory_count = self._count_trajectories(str(train_csv))
            if trajectory_count > 0:
                self.trajectory_spinbox.setRange(1, trajectory_count)
                self.trajectory_spinbox.setValue(1)
                self.show_btn.setEnabled(True)
                self.log(f"✓ Found {trajectory_count} trajectories in CSV")
            else:
                self.log("⚠️ No trajectories found in CSV")
                self.show_btn.setEnabled(False)
        else:
            self.log("⚠️ train.csv not found - trajectory selection disabled")
            self.show_btn.setEnabled(False)
    
    def _count_trajectories(self, csv_path: str) -> int:
        """Count the number of trajectories in the CSV file."""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Count lines (excluding header)
                return sum(1 for _ in f) - 1  # Subtract 1 for header
        except Exception as e:
            self.log(f"⚠️ Error counting trajectories: {e}")
            return 0
    
    def on_show_clicked(self):
        """Handle Show button click - load and display selected trajectory."""
        trajectory_num = int(self.trajectory_spinbox.value())
        self.load_trajectory(trajectory_num)
    
    def on_clear_clicked(self):
        """Handle Clear button click - clear all drawn items."""
        # Clear previous items
        for item in self._route_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                # Item already removed, ignore
                pass
        self._route_items = []
        self._bounding_box_polygon = None
        self._current_sumo_points = None
        # Also clear segments when clearing trajectory
        self.on_clear_segment_clicked()
        # Disable segment button when trajectory is cleared
        self.show_segment_btn.setEnabled(False)
        self.segment_spinbox.setRange(1, 1)
        self.log("✓ Cleared all trajectory items")
    
    def on_show_segment_clicked(self):
        """Handle Show Segment button click - draw selected segment in magenta."""
        if not self._current_sumo_points or len(self._current_sumo_points) < 2:
            self.log("⚠️ No trajectory loaded or not enough points")
            return
        
        segment_num = int(self.segment_spinbox.value())
        max_segments = len(self._current_sumo_points) - 1
        
        if segment_num < 1 or segment_num > max_segments:
            self.log(f"⚠️ Invalid segment number: {segment_num} (must be 1-{max_segments})")
            return
        
        # Clear previous segment
        self.on_clear_segment_clicked()
        
        # Draw the selected segment
        self._draw_segment(segment_num)
    
    def on_clear_segment_clicked(self):
        """Handle Clear Segment button click - clear segment items."""
        # Clear previous segment items
        for item in self._segment_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                # Item already removed, ignore
                pass
        self._segment_items = []
    
    def _draw_segment(self, segment_num: int):
        """Draw a specific segment (GPS points and connecting line) in magenta.
        
        Args:
            segment_num: Segment number (1-indexed), where 1 = GPS points 1-2, 2 = GPS points 2-3, etc.
        """
        if not self._current_sumo_points or len(self._current_sumo_points) < 2:
            return
        
        max_segments = len(self._current_sumo_points) - 1
        if segment_num < 1 or segment_num > max_segments:
            return
        
        # Segment N connects point N to point N+1 (0-indexed: point segment_num-1 to segment_num)
        point_idx1 = segment_num - 1  # First point of segment (0-indexed)
        point_idx2 = segment_num       # Second point of segment (0-indexed)
        
        if point_idx1 >= len(self._current_sumo_points) or point_idx2 >= len(self._current_sumo_points):
            return
        
        x1, y1 = self._current_sumo_points[point_idx1]
        x2, y2 = self._current_sumo_points[point_idx2]
        
        # Magenta color for segment
        magenta_color = QColor(255, 0, 255)  # Magenta
        magenta_pen = QPen(magenta_color, 5)  # Thicker line (5px) for visibility
        magenta_pen.setStyle(Qt.SolidLine)
        magenta_pen.setCapStyle(Qt.RoundCap)
        magenta_pen.setJoinStyle(Qt.RoundJoin)
        
        # Draw connecting line
        line = self.map_view.scene.addLine(x1, y1, x2, y2, magenta_pen)
        line.setZValue(400)  # Very high Z-value to be above everything
        self._segment_items.append(line)
        
        # Draw GPS points as circles (larger than stars for visibility)
        point_radius = 12
        point_pen = QPen(magenta_color, 3)
        point_brush = QBrush(magenta_color)
        
        # Point 1
        circle1 = self.map_view.scene.addEllipse(
            x1 - point_radius/2, y1 - point_radius/2,
            point_radius, point_radius,
            point_pen, point_brush
        )
        circle1.setZValue(401)  # Above the line
        self._segment_items.append(circle1)
        
        # Point 2
        circle2 = self.map_view.scene.addEllipse(
            x2 - point_radius/2, y2 - point_radius/2,
            point_radius, point_radius,
            point_pen, point_brush
        )
        circle2.setZValue(401)  # Above the line
        self._segment_items.append(circle2)
        
        self.log(f"✓ Drew segment {segment_num} in magenta (GPS points {point_idx1+1}→{point_idx2+1})")
    
    def load_trajectory(self, trajectory_num: int):
        """Load and display a specific trajectory with trimming.
        
        Args:
            trajectory_num: The trajectory number (1-indexed) to load
        """
        # Validate trajectory number
        if trajectory_num < 1:
            self.log(f"❌ Invalid trajectory number: {trajectory_num} (must be >= 1)")
            return
        
        # Get train_csv path
        train_csv = getattr(self, 'train_csv_path', None)
        
        if not train_csv or not train_csv.exists():
            # Fallback: try common locations (same as dataset_conversion_page)
            project_path = Path(self.project_path)
            
            # Try project_path/train.csv
            train_csv = project_path / "train.csv"
            
            # Try project_path/dataset/train.csv
            if not train_csv.exists():
                train_csv = project_path / "dataset" / "train.csv"
            
            # Try Porto/dataset/train.csv (use workspace root)
            if not train_csv.exists():
                workspace_root = _get_project_root()
                porto_train = workspace_root / 'Porto' / 'dataset' / 'train.csv'
                if porto_train.exists():
                    train_csv = porto_train
                else:
                    # Also try parent.parent in case project is in a subdirectory
                    porto_train = project_path.parent.parent / 'Porto' / 'dataset' / 'train.csv'
                    if porto_train.exists():
                        train_csv = porto_train
        
        if not train_csv or not train_csv.exists():
            self.log("❌ train.csv not found")
            self.log("  Checked: project_info.json, settings.json, train.csv, dataset/train.csv, Porto/dataset/train.csv")
            return
        
        # Validate trajectory number against available trajectories
        max_trajectories = self._count_trajectories(str(train_csv))
        if trajectory_num > max_trajectories:
            self.log(f"❌ Trajectory number {trajectory_num} exceeds available trajectories ({max_trajectories})")
            return
        
        self.log(f"Loading trajectory {trajectory_num}...")
        
        # Load trajectory
        try:
            polyline = self._load_trip_polyline(str(train_csv), trajectory_num)
            if not polyline:
                self.log(f"❌ Failed to load trajectory {trajectory_num}")
                return
            
            self.log(f"✓ Loaded trajectory {trajectory_num} with {len(polyline)} GPS points")
            
            # Apply trimming
            trimmed_polyline = self._apply_trimming(polyline)
            self.log(f"✓ After trimming: {len(trimmed_polyline)} GPS points")
            
            # Clear previous items before drawing new trajectory
            for item in self._route_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    # Item already removed, ignore
                    pass
            self._route_items = []
            self._bounding_box_polygon = None
            
            # Draw trajectory
            self._draw_trajectory(trimmed_polyline)
            self.log(f"✓ Trajectory {trajectory_num} drawn on map")
            
        except Exception as e:
            self.log(f"❌ Error loading trajectory {trajectory_num}: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_trip_polyline(self, csv_path: str, trip_num: int) -> List[List[float]]:
        """Load polyline for a specific trip from CSV (matches dataset_conversion_page format)."""
        import ast
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip header
                next(f, None)
                
                for i, line in enumerate(f, 1):
                    if i == trip_num:
                        # Parse the line to extract POLYLINE column (last column)
                        # CSV format: "TRIP_ID","CALL_TYPE",...,"POLYLINE"
                        try:
                            # Find the POLYLINE column (starts with "[[ and ends with ]]")
                            polyline_start = line.rfind('"[[')
                            if polyline_start == -1:
                                polyline_start = line.rfind('"[]')
                            
                            if polyline_start != -1:
                                polyline_str = line[polyline_start+1:].strip().rstrip('"')
                                polyline = ast.literal_eval(polyline_str)
                                return polyline
                        except Exception as e:
                            self.log(f"Error parsing polyline: {e}")
                            return []
        except Exception as e:
            self.log(f"Error reading CSV: {e}")
        
        return []
    
    def _detect_real_start_and_end(self, polyline: List[List[float]]) -> Tuple[int, int]:
        """
        Detect real start and end points by finding where static points end.
        Same logic as dataset_conversion_page.
        
        For taxi datasets:
        - Real start: The last point before movement begins (after static pickup points)
        - Real end: The first point at destination (before static dropoff points)
        
        Returns: (real_start_index, real_end_index)
        """
        if not polyline or len(polyline) < 3:
            return 0, len(polyline) - 1 if polyline else 0
        
        import math
        
        def haversine_distance(coord1: tuple, coord2: tuple) -> float:
            """Calculate distance between two GPS coordinates in meters."""
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            
            # Earth radius in meters
            R = 6371000
            
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
        
        # Threshold for considering points as "static" (in meters)
        STATIC_THRESHOLD = 15.0  # Points within 15 meters are considered static
        
        # Find real start: look for the first significant movement
        # The real start is the last point before movement begins
        # Example: points 0,1,2,3 are static, point 4 is distant -> real start is point 3 (index 3)
        real_start = 0
        for i in range(len(polyline) - 1):
            distance = haversine_distance(polyline[i], polyline[i + 1])
            if distance > STATIC_THRESHOLD:
                # Found first significant movement between i and i+1
                # Real start is the last static point, which is i (the point before the jump)
                real_start = i
                break
        
        # Find real end: look backwards for the last significant movement
        # The real end is the first point at the destination (after movement ends)
        # Example: movement ends between points i-1 and i, real end is point i
        real_end = len(polyline) - 1
        for i in range(len(polyline) - 1, 0, -1):
            distance = haversine_distance(polyline[i - 1], polyline[i])
            if distance > STATIC_THRESHOLD:
                # Found last significant movement between i-1 and i
                # Real end is the first point at destination, which is i
                real_end = i
                break
        
        return real_start, real_end
    
    def _apply_trimming(self, polyline: List[List[float]]) -> List[List[float]]:
        """Apply start/end trimming to find real start and destination.
        Uses the same logic as dataset_conversion_page._detect_real_start_and_end."""
        if len(polyline) < 2:
            return polyline
        
        # Detect real start and end using haversine distance threshold
        real_start_idx, real_end_idx = self._detect_real_start_and_end(polyline)
        
        # Return trimmed polyline
        trimmed = polyline[real_start_idx:real_end_idx + 1]
        self.log(f"  Trimmed: removed {real_start_idx} from start, {len(polyline) - real_end_idx - 1} from end")
        return trimmed
    
    def _draw_trajectory(self, polyline: List[List[float]]):
        """Draw GPS points and connecting line on map."""
        if not polyline or len(polyline) < 2:
            return
        
        # Clear previous items
        for item in self._route_items:
            self.map_view.scene.removeItem(item)
        self._route_items = []
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        # Convert GPS to SUMO coordinates (ORIGINAL SUMO coordinates, NOT Y-flipped)
        # We'll only Y-flip when drawing, not for calculations
        sumo_points_original = []  # Store in ORIGINAL SUMO coordinates
        for idx, (lon, lat) in enumerate(polyline):
            sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if sumo_coords:
                x, y = sumo_coords  # ORIGINAL SUMO coordinates
                sumo_points_original.append((x, y))
                
                # Debug: Log first 2 GPS points conversion
                if idx < 2:
                    self.log(f"  [GPS Conversion] Point {idx+1}: GPS=({lon:.6f}, {lat:.6f}) -> SUMO=({x:.1f}, {y:.1f}) [ORIGINAL, not Y-flipped]")
        
        # Convert to Y-flipped coordinates ONLY for drawing
        sumo_points = [(x, flip_y(y)) for x, y in sumo_points_original]
        
        if len(sumo_points) < 2:
            self.log("❌ Not enough valid SUMO coordinates")
            return
        
        # Draw connecting lines between points (50% thinner and dashed)
        line_color = QColor(0, 100, 255)  # Blue
        line_pen = QPen(line_color, 3)  # 50% thinner (was 6, now 3)
        line_pen.setStyle(Qt.DashLine)  # Dashed line
        for i in range(len(sumo_points) - 1):
            x1, y1 = sumo_points[i]
            x2, y2 = sumo_points[i + 1]
            
            line = self.map_view.scene.addLine(
                x1, y1, x2, y2,
                line_pen
            )
            line.setZValue(5)  # Above network, below points
            self._route_items.append(line)
        
        # Draw GPS points as stars (same as conversion page for alignment verification)
        from PySide6.QtGui import QPainterPath
        from PySide6.QtWidgets import QGraphicsPathItem
        
        def create_star_path(center_x: float, center_y: float, radius: float) -> QPainterPath:
            """Create a star shape path (same as conversion page)."""
            path = QPainterPath()
            # 5-pointed star
            num_points = 5
            outer_radius = radius
            inner_radius = radius * 0.4
            
            for i in range(num_points * 2):
                angle = (i * math.pi) / num_points - math.pi / 2  # Start at top
                if i % 2 == 0:
                    r = outer_radius
                else:
                    r = inner_radius
                
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            path.closeSubpath()
            return path
        
        # Star settings (same as conversion page)
        star_color = QColor(255, 255, 0)  # Yellow
        star_pen = QPen(star_color, 2.0)
        star_brush = QBrush(star_color)
        star_radius = 8.0
        
        # Draw GPS points as stars
        for idx, (x, y_flipped) in enumerate(sumo_points):
            # Create star path
            star_path = create_star_path(x, y_flipped, star_radius)
            
            # Create graphics item for star
            star_item = QGraphicsPathItem(star_path)
            star_item.setPen(star_pen)
            star_item.setBrush(star_brush)
            star_item.setZValue(200)  # Above network, below text
            self.map_view.scene.addItem(star_item)
            self._route_items.append(star_item)
            
            # Number label - white text with black border
            # Draw black outline by drawing text multiple times with offsets
            text_str = str(idx + 1)
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            
            # Create white text item
            text_item = QGraphicsTextItem(text_str)
            text_item.setPos(x + 8, y_flipped - 8)
            text_item.setDefaultTextColor(QColor(255, 255, 255))  # White text
            text_item.setFont(font)
            text_item.setZValue(300)
            
            # Add black outline by drawing black text behind with slight offsets
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dx, dy in offsets:
                outline_item = QGraphicsTextItem(text_str)
                outline_item.setPos(x + 8 + dx, y_flipped - 8 + dy)
                outline_item.setDefaultTextColor(QColor(0, 0, 0))  # Black outline
                outline_item.setFont(font)
                outline_item.setZValue(299)  # Behind white text
                self.map_view.scene.addItem(outline_item)
                self._route_items.append(outline_item)
            
            # Add white text on top
            self.map_view.scene.addItem(text_item)
            self._route_items.append(text_item)
        
        # Draw minimum bounding box with padding
        # Create bounding box from ORIGINAL SUMO coordinates (for calculations)
        # Then Y-flip for display
        self._draw_bounding_box(sumo_points_original, sumo_points)
        
        # Calculate and log normalized coordinates relative to bounding box
        # if self._bounding_box_params:
        #     self._calculate_and_log_normalized_coordinates(sumo_points_original)
        
        # Store current trajectory points for segment selection
        self._current_sumo_points = sumo_points
        
        # Update segment spinbox range (number of segments = number of points - 1)
        if len(sumo_points) > 1:
            max_segments = len(sumo_points) - 1
            self.segment_spinbox.setRange(1, max_segments)
            self.segment_spinbox.setValue(1)
            self.show_segment_btn.setEnabled(True)
            self.log(f"  Segment selection enabled: {max_segments} segments available")
        else:
            self.show_segment_btn.setEnabled(False)
        
        # Fit view to show entire trajectory
        if sumo_points:
            from PySide6.QtCore import QRectF
            xs = [x for x, _ in sumo_points]
            ys = [y for _, y in sumo_points]
            rect = QRectF(min(xs) - 100, min(ys) - 100, 
                         max(xs) - min(xs) + 200, max(ys) - min(ys) + 200)
            self.map_view.fitInView(rect, Qt.KeepAspectRatio)
        
        self.log(f"✓ Drew {len(sumo_points)} GPS points and connecting line")
    
    def _draw_bounding_box(self, sumo_points_original: List[Tuple[float, float]], sumo_points_flipped: List[Tuple[float, float]]):
        """
        Draw a rotated minimum bounding box around all GPS points with padding.
        
        Args:
            sumo_points_original: GPS points in ORIGINAL SUMO coordinates (for calculations)
            sumo_points_flipped: GPS points in Y-flipped SUMO coordinates (for display)
        """
        if len(sumo_points_original) < 2:
            self.log("⚠️ Not enough points for bounding box")
            return
        
        # Log point range for debugging (ORIGINAL SUMO)
        xs = [x for x, y in sumo_points_original]
        ys = [y for x, y in sumo_points_original]
        self.log(f"  Point range (ORIGINAL SUMO): x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")
        
        # Find minimum bounding box (rotated rectangle) using ORIGINAL SUMO coordinates
        min_box = self._find_minimum_bounding_box(sumo_points_original, padding_meters=200.0)
        
        if not min_box:
            self.log("⚠️ Failed to calculate bounding box")
            return
        
        # Extract box parameters
        center_x, center_y, width, height, angle = min_box
        
        # Store bounding box parameters for normalized coordinate calculation
        self._bounding_box_params = {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'angle': angle
        }
        
        self.log(f"📦 Bounding box: center=({center_x:.1f}, {center_y:.1f}), size=({width:.1f}, {height:.1f}), angle={math.degrees(angle):.1f}°")
        
        # Create rectangle corners (before rotation) - centered at origin
        half_width = width / 2
        half_height = height / 2
        
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # Rotate corners around center
        # IMPORTANT: We rotated points by +angle to find the box,
        # so we need to rotate the rectangle by -angle to get back to original coordinates
        rotated_corners = []
        cos_a = math.cos(-angle)  # Negative angle to rotate back
        sin_a = math.sin(-angle)
        
        for dx, dy in corners:
            # Rotate point (clockwise rotation to undo the counterclockwise rotation)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            # Translate to center
            rotated_corners.append(QPointF(center_x + rx, center_y + ry))
        
        # Verify all points are inside the box using point-in-polygon test (ORIGINAL SUMO)
        self._verify_points_in_box(sumo_points_original, rotated_corners)
        
        # Ensure polygon is closed (add first point at end if needed)
        if len(rotated_corners) > 0 and rotated_corners[0] != rotated_corners[-1]:
            rotated_corners.append(rotated_corners[0])
        
        # Create polygon in ORIGINAL SUMO coordinates (for calculations)
        from PySide6.QtGui import QPolygonF
        polygon_original = QPolygonF(rotated_corners)
        
        # Log corner coordinates for debugging (ORIGINAL SUMO)
        corner_strs = [f"({c.x():.1f}, {c.y():.1f})" for c in rotated_corners[:4]]  # First 4 corners
        self.log(f"  Box corners (ORIGINAL SUMO): {', '.join(corner_strs)}")
        
        # Verify polygon is valid
        if polygon_original.isEmpty():
            self.log("⚠️ Polygon is empty!")
            return
        
        # Store polygon for edge intersection checks (in ORIGINAL SUMO coordinates)
        self._bounding_box_polygon = polygon_original
        
        # Y-flip polygon corners for display
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            return y_max + y_min - y
        
        flipped_corners = []
        for corner in rotated_corners:
            # Y-flip only for display
            flipped_corners.append(QPointF(corner.x(), flip_y(corner.y())))
        
        # Create flipped polygon for display
        flipped_polygon = QPolygonF(flipped_corners)
        
        # Draw rectangle with black border (thicker pen for visibility) - Y-flipped for display
        rect_item = QGraphicsPolygonItem(flipped_polygon)
        black_pen = QPen(QColor(0, 0, 0), 4)  # Black border, 4px width for visibility
        black_pen.setStyle(Qt.SolidLine)
        black_pen.setCapStyle(Qt.RoundCap)
        black_pen.setJoinStyle(Qt.RoundJoin)
        rect_item.setPen(black_pen)
        rect_item.setBrush(QBrush(Qt.NoBrush))  # No fill
        rect_item.setZValue(50)  # Above trajectory lines (which are at 5) but below points (which are at 200)
        
        # Verify item is valid before adding
        if rect_item.boundingRect().isEmpty():
            self.log("⚠️ Rectangle bounding rect is empty!")
            return
        
        self.map_view.scene.addItem(rect_item)
        self._route_items.append(rect_item)
        self.log(f"✓ Bounding box drawn: {len(rotated_corners)} corners, z-value={rect_item.zValue()}, bbox={rect_item.boundingRect()}")
        
        # Color vehicle edges inside the bounding box in red
        self._color_edges_in_box()
    
    def _color_edges_in_box(self):
        """Color vehicle-allowed edges inside the bounding box polygon in red."""
        if not self._bounding_box_polygon:
            self.log("⚠️ Bounding box polygon not set, cannot color edges")
            return
        
        if not self.network_parser:
            self.log("⚠️ Network parser not loaded")
            return
        
        edges = self.network_parser.get_edges()
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        # Red pen for vehicle edges inside bounding box
        red_pen = QPen(QColor(255, 0, 0), 3)  # Red, 3px width
        red_pen.setStyle(Qt.SolidLine)
        
        edges_in_box = []
        edges_checked = 0
        
        for edge_id, edge_data in edges.items():
            # Only check vehicle-allowed edges
            if not edge_data.get('allows_passenger', True):
                continue
            
            if not edge_data.get('lanes'):
                continue
            
            first_lane = edge_data['lanes'][0]
            shape_points = first_lane.get('shape', [])
            if len(shape_points) < 2:
                continue
            
            edges_checked += 1
            
            # Check if edge is inside bounding box
            # ALL in ORIGINAL SUMO coordinates: shape_points, bounding_box_polygon
            # NO Y-flip needed - everything is in ORIGINAL SUMO coordinates
            edge_in_box = False
            
            # Check if any point of the edge is inside the polygon
            for point in shape_points:
                x, y = point[0], point[1]  # ORIGINAL SUMO coordinates
                if self._bounding_box_polygon.containsPoint(QPointF(x, y), Qt.OddEvenFill):
                    edge_in_box = True
                    break
            
            # If no point is inside, check if any segment intersects the polygon
            if not edge_in_box:
                for i in range(len(shape_points) - 1):
                    x1, y1 = shape_points[i][0], shape_points[i][1]  # ORIGINAL SUMO
                    x2, y2 = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL SUMO
                    if self._line_intersects_polygon(QPointF(x1, y1), QPointF(x2, y2), self._bounding_box_polygon):
                        edge_in_box = True
                        break
            
            if edge_in_box:
                edges_in_box.append((edge_id, shape_points))
        
        self.log(f"🔴 Found {len(edges_in_box)} vehicle edges inside bounding box (checked {edges_checked} edges)")
        
        # Draw edges in red (Y-flip for display)
        for edge_id, shape_points in edges_in_box:
            for i in range(len(shape_points) - 1):
                x1, y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL SUMO
                x2, y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL SUMO
                
                # Y-flip ONLY for display
                y1 = flip_y(y1_sumo)
                y2 = flip_y(y2_sumo)
                
                line = self.map_view.scene.addLine(x1, y1, x2, y2, red_pen)
                line.setZValue(15)  # Above network edges (0) but below other overlays
                self._route_items.append(line)
        
        self.log(f"✓ Drew {len(edges_in_box)} vehicle edges in red inside bounding box")
    
    def _calculate_and_log_normalized_coordinates(self, sumo_points_original: List[Tuple[float, float]]):
        """Calculate and log normalized coordinates of polyline relative to bounding box.
        
        The transformation formula:
        1. Translate: (x', y') = (x - center_x, y - center_y)
        2. Rotate by -angle: (x'', y'') = (x'*cos(-angle) - y'*sin(-angle), x'*sin(-angle) + y'*cos(-angle))
        3. Normalize: (x_norm, y_norm) = (x'' / (width/2), y'' / (height/2))
        
        Result: Normalized coordinates in range [-1, 1] relative to bounding box
        where (-1, -1) is bottom-left, (1, 1) is top-right, (0, 0) is center.
        """
        if not self._bounding_box_params:
            return
        
        params = self._bounding_box_params
        center_x = params['center_x']
        center_y = params['center_y']
        width = params['width']
        height = params['height']
        angle = params['angle']
        
        # Log the transformation formula
        self.log("=" * 60)
        self.log("📐 NORMALIZED COORDINATES FORMULA (relative to bounding box)")
        self.log("=" * 60)
        self.log(f"Bounding box parameters:")
        self.log(f"  Center: ({center_x:.3f}, {center_y:.3f})")
        self.log(f"  Size: {width:.3f} x {height:.3f}")
        self.log(f"  Angle: {math.degrees(angle):.3f}° ({angle:.6f} radians)")
        self.log("")
        self.log("Transformation steps:")
        self.log("  1. Translate: (x', y') = (x - {:.3f}, y - {:.3f})".format(center_x, center_y))
        self.log("  2. Rotate by -{:.3f}°: (x'', y'') = (x'*cos(-θ) - y'*sin(-θ), x'*sin(-θ) + y'*cos(-θ))".format(math.degrees(angle)))
        self.log("  3. Normalize: (x_norm, y_norm) = (x'' / {:.3f}, y'' / {:.3f})".format(width/2, height/2))
        self.log("")
        self.log("Normalized coordinates range: [-1, 1]")
        self.log("  (-1, -1) = bottom-left corner")
        self.log("  (1, 1) = top-right corner")
        self.log("  (0, 0) = center")
        self.log("")
        
        # Calculate normalized coordinates for each point
        cos_neg_angle = math.cos(-angle)
        sin_neg_angle = math.sin(-angle)
        half_width = width / 2
        half_height = height / 2
        
        normalized_points = []
        self.log("Normalized polyline coordinates:")
        self.log("  [")
        
        for idx, (x, y) in enumerate(sumo_points_original):
            # Step 1: Translate to center at origin
            x_translated = x - center_x
            y_translated = y - center_y
            
            # Step 2: Rotate by -angle
            x_rotated = x_translated * cos_neg_angle - y_translated * sin_neg_angle
            y_rotated = x_translated * sin_neg_angle + y_translated * cos_neg_angle
            
            # Step 3: Normalize by box dimensions
            x_norm = x_rotated / half_width
            y_norm = y_rotated / half_height
            
            normalized_points.append((x_norm, y_norm))
            
            # Log first 5 and last 5 points, plus every 10th point in between
            if idx < 5 or idx >= len(sumo_points_original) - 5 or idx % 10 == 0:
                self.log(f"    [{x_norm:.6f}, {y_norm:.6f}],  # Point {idx+1}: original=({x:.2f}, {y:.2f})")
        
        if len(sumo_points_original) > 10:
            self.log(f"    ... ({len(sumo_points_original) - 10} more points) ...")
        
        self.log("  ]")
        self.log("")
        self.log(f"Total points: {len(normalized_points)}")
        
        # Log coordinate ranges
        x_norms = [p[0] for p in normalized_points]
        y_norms = [p[1] for p in normalized_points]
        self.log(f"Normalized X range: [{min(x_norms):.6f}, {max(x_norms):.6f}]")
        self.log(f"Normalized Y range: [{min(y_norms):.6f}, {max(y_norms):.6f}]")
        self.log("=" * 60)
    
    def _line_intersects_polygon(self, p1: QPointF, p2: QPointF, polygon) -> bool:
        """Check if a line segment intersects a polygon."""
        # Check intersection with each polygon edge
        polygon_points = []
        # QPolygonF can be accessed like a list or using at() method
        for i in range(polygon.size()):
            polygon_points.append(polygon.at(i))
        
        for i in range(len(polygon_points)):
            p3 = polygon_points[i]
            p4 = polygon_points[(i + 1) % len(polygon_points)]
            
            if self._line_segments_intersect(p1, p2, p3, p4):
                return True
        
        return False
    
    def _line_segments_intersect(self, p1: QPointF, p2: QPointF, p3: QPointF, p4: QPointF) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            """Check if three points are in counterclockwise order."""
            return (C.y() - A.y()) * (B.x() - A.x()) > (B.y() - A.y()) * (C.x() - A.x())
        
        # Check if line segments intersect using cross product method
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and 
                ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    def _verify_points_in_box(self, points: List[Tuple[float, float]], box_corners: List[QPointF]):
        """Verify that all points are inside the bounding box."""
        if len(box_corners) < 4:
            return
        
        # Check if points are inside using point-in-polygon test
        outside_points = []
        for x, y in points:
            point = QPointF(x, y)
            # Simple point-in-polygon test (ray casting)
            inside = False
            j = len(box_corners) - 1
            for i in range(len(box_corners)):
                xi, yi = box_corners[i].x(), box_corners[i].y()
                xj, yj = box_corners[j].x(), box_corners[j].y()
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            
            if not inside:
                outside_points.append((x, y))
        
        if outside_points:
            self.log(f"⚠️ Warning: {len(outside_points)} points are outside the bounding box!")
            self.log(f"  First few outside points: {outside_points[:3]}")
        else:
            self.log(f"✓ All {len(points)} points are inside the bounding box")
    
    def _find_minimum_bounding_box(self, points: List[Tuple[float, float]], padding_meters: float = 200.0) -> Tuple[float, float, float, float, float]:
        """
        Find minimum bounding box (rotated rectangle) that contains all points.
        
        Uses rotation algorithm to find the angle that minimizes the bounding box area.
        
        Args:
            points: List of (x, y) tuples in SUMO coordinates
            padding_meters: Padding in meters to add to the box
        
        Returns:
            Tuple of (center_x, center_y, width, height, angle_radians) or None
        """
        if len(points) < 2:
            return None
        
        # Estimate meters to SUMO coordinate units
        # Calculate conversion factor by comparing SUMO distance to GPS distance
        meters_per_sumo_unit = 1.0  # Default assumption
        
        if len(points) >= 2 and self.network_parser:
            try:
                # Get Y bounds for flipping
                y_min = getattr(self.map_view, '_network_y_min', 0)
                y_max = getattr(self.map_view, '_network_y_max', 0)
                
                def flip_y(y):
                    return y_max + y_min - y
                
                # Convert SUMO coordinates back to GPS to calculate real distance
                # Use first two points
                x1_sumo, y1_sumo_flipped = points[0]
                x2_sumo, y2_sumo_flipped = points[1]
                
                # Unflip Y coordinates
                y1_sumo = flip_y(y1_sumo_flipped)
                y2_sumo = flip_y(y2_sumo_flipped)
                
                # Convert back to GPS (if network parser supports it)
                # For now, estimate: 1 degree latitude ≈ 111,000 meters
                # SUMO networks typically use meters or close to meters
                # Calculate SUMO distance
                sumo_dist = math.sqrt((x2_sumo - x1_sumo)**2 + (y2_sumo - y1_sumo)**2)
                
                # If distance is reasonable (not too small), use it
                # Otherwise assume 1:1 mapping
                if sumo_dist > 10:  # If distance is more than 10 SUMO units
                    # Estimate: for Porto area, typical GPS-to-SUMO conversion
                    # is approximately 1:1 (meters), but can vary
                    # Use a conservative estimate
                    meters_per_sumo_unit = 1.0
            except Exception:
                # Fallback to 1:1 assumption
                meters_per_sumo_unit = 1.0
        
        padding = padding_meters * meters_per_sumo_unit
        
        # Try different rotation angles to find minimum area
        min_area = float('inf')
        best_box = None
        
        # Try angles from 0 to 180 degrees (need full rotation to find minimum)
        for angle_deg in range(0, 181, 1):  # Step by 1 degree
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Calculate centroid of points for better rotation center
            cx = sum(x for x, y in points) / len(points)
            cy = sum(y for x, y in points) / len(points)
            
            # Rotate all points around centroid
            rotated_points = []
            for x, y in points:
                # Translate to centroid
                dx = x - cx
                dy = y - cy
                # Rotate (counterclockwise)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rotated_points.append((rx, ry))
            
            # Find axis-aligned bounding box of rotated points
            xs = [p[0] for p in rotated_points]
            ys = [p[1] for p in rotated_points]
            
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            
            # Add padding to each side
            width = max_x - min_x + 2 * padding
            height = max_y - min_y + 2 * padding
            
            # The center of the bounding box in rotated coordinates is the centroid
            # Since we rotated around centroid, the center maps back to centroid
            # But we need to account for the fact that padding expands the box symmetrically
            # The center in rotated coords is still (0, 0) relative to centroid
            # So center in original coords is just the centroid
            center_x = cx
            center_y = cy
            
            area = width * height
            
            if area < min_area:
                min_area = area
                # Store the box - center is centroid, width/height include padding
                best_box = (center_x, center_y, width, height, angle_rad)
        
        # If best_box is None or has invalid dimensions, fall back to axis-aligned box
        if best_box is None or best_box[2] <= 0 or best_box[3] <= 0:
            # Fallback: simple axis-aligned bounding box
            xs = [x for x, y in points]
            ys = [y for x, y in points]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            width = max(xs) - min(xs) + 2 * padding
            height = max(ys) - min(ys) + 2 * padding
            best_box = (cx, cy, width, height, 0.0)  # No rotation
        
        return best_box
