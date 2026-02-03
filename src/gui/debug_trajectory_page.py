"""
Debug page for visualizing the first trajectory with trimming applied.
Simple page with just map, network, GPS points, and connecting line.
"""

import copy
import csv
import gzip
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (QDoubleSpinBox, QGraphicsEllipseItem,
                               QGraphicsPolygonItem, QGraphicsTextItem,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.project_manager import _get_project_root
from src.utils.route_finding import apply_trimming as route_apply_trimming
from src.utils.route_finding import (build_edges_data, build_node_positions,
                                     compute_green_orange_edges,
                                     project_point_onto_polyline,
                                     shortest_path_dijkstra)
from src.utils.trip_validator import (DEFAULT_MAX_SEGMENT_DISTANCE,
                                      TripValidationResult,
                                      split_at_invalid_segments,
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
        self._green_edge_items = []  # Store green edge items for clearing
        self._red_edges_data = []  # Store red edges data: list of (edge_id, edge_data, shape_points) tuples
        self._route_path_items = []  # Computed shortest path lines (red) for clearing
        self.train_csv_path = None  # To store the resolved train.csv path
        self._bounding_box_polygon = None  # Store bounding box polygon for edge intersection checks
        self._bounding_box_params = None  # Store bounding box parameters (center_x, center_y, width, height, angle)
        self._current_sumo_points = None  # Store current trajectory points (Y-flipped for display)
        self._current_sumo_segments = []  # Per-segment sumo points (Y-flipped) for route finding
        self._network_file_path = None  # Path to loaded .net.xml for export
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
        
        footer_layout.addSpacing(10)
        
        # Process All Segments button
        self.process_all_segments_btn = QPushButton("Process All Segments")
        self.process_all_segments_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.process_all_segments_btn.clicked.connect(self.on_process_all_segments_clicked)
        self.process_all_segments_btn.setEnabled(False)  # Disabled until trajectory is loaded
        footer_layout.addWidget(self.process_all_segments_btn)
        
        # Export Bounding Box Network button
        self.export_bbox_network_btn = QPushButton("Export Bounding Box Network")
        self.export_bbox_network_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.export_bbox_network_btn.setToolTip("Export edges inside the bounding box to trajectory_<N>.net.xml (loadable SUMO network)")
        self.export_bbox_network_btn.clicked.connect(self.on_export_bbox_network_clicked)
        self.export_bbox_network_btn.setEnabled(False)
        footer_layout.addWidget(self.export_bbox_network_btn)
        
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
        self._network_file_path = Path(network_file)
        
        # Load network parser
        try:
            self.network_parser = NetworkParser(str(network_file))
            self.map_view.load_network(self.network_parser, roads_junctions_only=False)
            
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
        # No OSM map tiles - network only
        self.map_view.set_osm_map_visible(False)
        
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
        self._red_edges_data = []  # Clear red edges data
        # Also clear segments when clearing trajectory
        self.on_clear_segment_clicked()
        # Disable segment and export buttons when trajectory is cleared
        self.show_segment_btn.setEnabled(False)
        self.process_all_segments_btn.setEnabled(False)
        if getattr(self, 'export_bbox_network_btn', None) is not None:
            self.export_bbox_network_btn.setEnabled(False)
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
        """Handle Clear Segment button click - clear segment items and green edges."""
        # Clear previous segment items
        for item in self._segment_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                # Item already removed, ignore
                pass
        self._segment_items = []
        
        # Clear green edge items (orange and green edges, but NOT red edges)
        for item in self._green_edge_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                # Item already removed, ignore
                pass
        self._green_edge_items = []
    
    def on_process_all_segments_clicked(self):
        """Handle Process All Segments button click - run route finding (all edges, green/orange + Dijkstra/A*)."""
        if not self._current_sumo_segments:
            self.log("⚠️ No trajectory loaded or no segments (load trajectory first)")
            return
        if not self.network_parser:
            self.log("⚠️ Network not loaded")
            return
        self.log("🔄 Running route finding (all edges, same logic as view_network)...")
        self._run_route_finding()
    
    def on_export_bbox_network_clicked(self):
        """Export the network inside the bounding box to trajectory_<N>.net.xml (loadable SUMO network)."""
        if not self._red_edges_data:
            self.log("⚠️ No edges in bounding box. Draw trajectory and bounding box first.")
            return
        if not getattr(self, '_network_file_path', None) or not self._network_file_path.exists():
            self.log("⚠️ No source network file path. Reload the page.")
            return
        trajectory_num = int(self.trajectory_spinbox.value())
        project_path = Path(self.project_path)
        out_dir = project_path / "config"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"trajectory_{trajectory_num}.net.xml"
        
        # Edge base IDs and node IDs (from/to) for edges in box
        edge_base_ids: Set[str] = set()
        node_ids: Set[str] = set()
        for edge_id, edge_data, _ in self._red_edges_data:
            base_id = edge_id.split("#")[0] if "#" in edge_id else edge_id
            edge_base_ids.add(base_id)
            fn, tn = edge_data.get("from"), edge_data.get("to")
            if fn:
                node_ids.add(fn)
            if tn:
                node_ids.add(tn)
        
        # Parse source .net.xml (support gzip)
        try:
            if str(self._network_file_path).endswith(".gz") or self._network_file_path.suffix == ".gz":
                with gzip.open(self._network_file_path, "rb") as f:
                    tree = ET.parse(f)
            else:
                tree = ET.parse(self._network_file_path)
            root = tree.getroot()
        except Exception as e:
            self.log(f"❌ Failed to read source network: {e}")
            return
        
        # Build new net: same root tag/attrib, location, then edges, then junctions
        ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
        new_root = ET.Element(root.tag, root.attrib)
        location = root.find("location")
        if location is not None:
            new_root.append(copy.deepcopy(location))
        for edge in root.findall("edge"):
            eid = edge.get("id")
            base_id = eid.split("#")[0] if eid and "#" in eid else (eid or "")
            if base_id in edge_base_ids:
                new_root.append(copy.deepcopy(edge))
        for junction in root.findall("junction"):
            jid = junction.get("id")
            if jid in node_ids:
                new_root.append(copy.deepcopy(junction))
        
        # Write output
        try:
            try:
                ET.indent(new_root, space="  ")
            except AttributeError:
                pass  # Python < 3.9
            out_tree = ET.ElementTree(new_root)
            out_tree.write(
                out_file,
                encoding="utf-8",
                default_namespace=None,
                xml_declaration=True,
                method="xml",
            )
        except Exception as e:
            self.log(f"❌ Failed to write {out_file.name}: {e}")
            return
        
        self.log(f"✓ Exported to {out_file}")
        
        # Validate: no missing data (every edge has from/to and lanes)
        try:
            parser = NetworkParser(str(out_file))
            exp_edges = parser.get_edges()
            missing_from_to = []
            missing_lanes = []
            for eid, ed in exp_edges.items():
                if not ed.get("from") or not ed.get("to"):
                    missing_from_to.append(eid)
                if not ed.get("lanes") or len(ed.get("lanes", [])) < 1:
                    missing_lanes.append(eid)
            if missing_from_to or missing_lanes:
                self.log(f"⚠️ Validation: {len(missing_from_to)} edge(s) missing from/to, {len(missing_lanes)} missing lanes")
                if missing_from_to:
                    self.log(f"   Missing from/to: {missing_from_to[:5]}{'...' if len(missing_from_to) > 5 else ''}")
                if missing_lanes:
                    self.log(f"   Missing lanes: {missing_lanes[:5]}{'...' if len(missing_lanes) > 5 else ''}")
            else:
                self.log(f"✓ Validation: {len(exp_edges)} edges, all have from/to and lanes (no missing data)")
        except Exception as e:
            self.log(f"⚠️ Could not validate exported file: {e}")
    
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
        
        # Calculate segment length in meters
        # Convert from Y-flipped display coordinates to original SUMO coordinates
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        # Unflip Y coordinates to get original SUMO coordinates
        y1_sumo = flip_y(y1)
        y2_sumo = flip_y(y2)
        x1_sumo = x1  # X doesn't need flipping
        x2_sumo = x2  # X doesn't need flipping
        
        # Calculate Euclidean distance in SUMO coordinates (assumed to be in meters)
        dx = x2_sumo - x1_sumo
        dy = y2_sumo - y1_sumo  # Use unflipped coordinates for distance
        segment_length = math.sqrt(dx * dx + dy * dy)
        
        self.log(f"✓ Drew segment {segment_num} in magenta (GPS points {point_idx1+1}→{point_idx2+1}, length={segment_length:.2f}m)")
        
        # Skip edge matching for very short segments (< 5m) - likely lane changes or GPS noise
        min_segment_length = 5.0  # meters
        if segment_length < min_segment_length:
            self.log(f"⏭️  Skipping edge matching for segment {segment_num} (length {segment_length:.2f}m < {min_segment_length}m threshold)")
            return
        
        # Find and color closest edges with matching direction
        # For first segment: also check closest edge to point 1
        # For last segment: also check closest edge to point 2 (last point)
        is_first_segment = (segment_num == 1)
        is_last_segment = (segment_num == max_segments)
        self._find_and_color_matching_edges(x1, y1, x2, y2, is_first_segment, is_last_segment)
    
    def _point_to_polyline_distance(self, point_x: float, point_y: float, polyline: List[List[float]]) -> float:
        """Calculate minimum distance from a point to a polyline.
        
        Args:
            point_x, point_y: Point coordinates
            polyline: List of [x, y] points forming the polyline
        
        Returns:
            Minimum distance from point to any segment of the polyline
        """
        if len(polyline) < 2:
            # If polyline has less than 2 points, calculate distance to the single point
            if len(polyline) == 1:
                dx = point_x - polyline[0][0]
                dy = point_y - polyline[0][1]
                return math.sqrt(dx * dx + dy * dy)
            return float('inf')
        
        min_distance = float('inf')
        
        # Check distance to each segment of the polyline
        for i in range(len(polyline) - 1):
            x1, y1 = polyline[i][0], polyline[i][1]
            x2, y2 = polyline[i+1][0], polyline[i+1][1]
            
            # Calculate distance from point to line segment
            # Vector from segment start to end
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy
            
            if seg_length_sq == 0:
                # Degenerate segment (start == end), use point-to-point distance
                dx = point_x - x1
                dy = point_y - y1
                dist = math.sqrt(dx * dx + dy * dy)
            else:
                # Vector from segment start to point
                point_dx = point_x - x1
                point_dy = point_y - y1
                
                # Project point onto segment line
                t = (point_dx * seg_dx + point_dy * seg_dy) / seg_length_sq
                
                # Clamp t to [0, 1] to stay on segment
                t = max(0.0, min(1.0, t))
                
                # Find closest point on segment
                closest_x = x1 + t * seg_dx
                closest_y = y1 + t * seg_dy
                
                # Calculate distance
                dx = point_x - closest_x
                dy = point_y - closest_y
                dist = math.sqrt(dx * dx + dy * dy)
            
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def _find_and_color_matching_edges(self, seg_x1: float, seg_y1: float, seg_x2: float, seg_y2: float,
                                       is_first_segment: bool = False, is_last_segment: bool = False,
                                       show_green_edges: bool = True):
        """Find and color the closest edges to segment start and end points.
        
        For each segment, finds:
        - The closest edge to the start point (point 1)
        - The closest edge to the end point (point 2)
        Both edges are colored orange (can be the same edge).
        Optionally displays top 5 segment-matching edges in green (excluding orange ones).
        
        Args:
            seg_x1, seg_y1: First GPS point of segment (Y-flipped for display)
            seg_x2, seg_y2: Second GPS point of segment (Y-flipped for display)
            is_first_segment: Unused (kept for compatibility)
            is_last_segment: Unused (kept for compatibility)
            show_green_edges: If True, also draw green edges (top 5 matches). Default True.
        """
        if not self._red_edges_data:
            self.log("⚠️ No red edges available (bounding box not drawn yet)")
            return
        
        # Get Y-flip function for coordinate conversion
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        # Calculate segment direction (already in Y-flipped coordinates)
        dx_seg = seg_x2 - seg_x1
        dy_seg = seg_y2 - seg_y1
        segment_angle = math.atan2(dy_seg, dx_seg)  # Range: [-π, π]
        
        # Calculate segment midpoint (Y-flipped coordinates)
        seg_mid_x = (seg_x1 + seg_x2) / 2
        seg_mid_y = (seg_y1 + seg_y2) / 2
        
        # Direction threshold: 45 degrees (π/4 radians)
        direction_threshold = math.pi / 4
        
        matching_edges = []
        
        # Check each red edge
        for edge_id, edge_data, shape_points in self._red_edges_data:
            # shape_points are in ORIGINAL SUMO coordinates
            if len(shape_points) < 2:
                continue
            
            # Calculate edge direction using first and last points
            # Get points in ORIGINAL SUMO coordinates
            x_start_sumo, y_start_sumo = shape_points[0][0], shape_points[0][1]
            x_end_sumo, y_end_sumo = shape_points[-1][0], shape_points[-1][1]
            
            # Y-flip for direction calculation to match segment coordinate system
            y_start_flipped = flip_y(y_start_sumo)
            y_end_flipped = flip_y(y_end_sumo)
            
            # Calculate edge direction
            dx_edge = x_end_sumo - x_start_sumo
            dy_edge = y_end_flipped - y_start_flipped
            edge_angle = math.atan2(dy_edge, dx_edge)  # Range: [-π, π]
            
            # Calculate angular difference (handle wrap-around)
            angle_diff = abs(segment_angle - edge_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # Check if direction matches (within threshold)
            if angle_diff > direction_threshold:
                continue
            
            # Convert GPS segment points from Y-flipped display coordinates to original SUMO coordinates
            # seg_x1, seg_y1, seg_x2, seg_y2 are in Y-flipped coordinates
            # Unflip Y coordinates to get original SUMO coordinates
            seg_y1_sumo = flip_y(seg_y1)  # Unflip: flip_y(flip_y(y)) = y
            seg_y2_sumo = flip_y(seg_y2)
            # X coordinates don't need flipping
            seg_x1_sumo = seg_x1
            seg_x2_sumo = seg_x2
            
            # Calculate minimum distance from GPS segment points to edge
            # Distance from point 1 to edge
            dist1 = self._point_to_polyline_distance(seg_x1_sumo, seg_y1_sumo, shape_points)
            # Distance from point 2 to edge
            dist2 = self._point_to_polyline_distance(seg_x2_sumo, seg_y2_sumo, shape_points)
            
            # Take the minimum distance
            distance = min(dist1, dist2)
            
            # Calculate combined score: distance + weighted angle difference
            # Weight factor: 20 meters per radian (so 1 radian ≈ 57° difference is equivalent to 20m distance)
            # Lower weight gives more importance to distance
            angle_weight = 5.0  # meters per radian
            combined_score = distance + angle_weight * angle_diff
            
            matching_edges.append((edge_id, edge_data, shape_points, distance, angle_diff, combined_score))
        
        # Convert GPS segment points from Y-flipped display coordinates to original SUMO coordinates
        seg_y1_sumo = flip_y(seg_y1)
        seg_y2_sumo = flip_y(seg_y2)
        seg_x1_sumo = seg_x1
        seg_x2_sumo = seg_x2
        
        # Find closest edge to start point (point 1)
        closest_to_point1 = None
        closest_to_point1_distance = float('inf')
        
        # Find closest edge to end point (point 2)
        closest_to_point2 = None
        closest_to_point2_distance = float('inf')
        
        # Check each red edge for closest to both points
        for edge_id, edge_data, shape_points in self._red_edges_data:
            if len(shape_points) < 2:
                continue
            
            # Calculate edge direction for angle check
            x_start_sumo, y_start_sumo = shape_points[0][0], shape_points[0][1]
            x_end_sumo, y_end_sumo = shape_points[-1][0], shape_points[-1][1]
            y_start_flipped = flip_y(y_start_sumo)
            y_end_flipped = flip_y(y_end_sumo)
            dx_edge = x_end_sumo - x_start_sumo
            dy_edge = y_end_flipped - y_start_flipped
            edge_angle = math.atan2(dy_edge, dx_edge)
            
            # Calculate angular difference with segment direction
            angle_diff = abs(segment_angle - edge_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # Check direction match (must be within threshold)
            if angle_diff > direction_threshold:
                continue
            
            # Calculate distance from point 1 to edge
            dist1 = self._point_to_polyline_distance(seg_x1_sumo, seg_y1_sumo, shape_points)
            if dist1 < closest_to_point1_distance:
                closest_to_point1_distance = dist1
                closest_to_point1 = (edge_id, edge_data, shape_points, dist1, angle_diff)
            
            # Calculate distance from point 2 to edge
            dist2 = self._point_to_polyline_distance(seg_x2_sumo, seg_y2_sumo, shape_points)
            if dist2 < closest_to_point2_distance:
                closest_to_point2_distance = dist2
                closest_to_point2 = (edge_id, edge_data, shape_points, dist2, angle_diff)
        
        # Determine orange edges (closest to point 1 and point 2)
        orange_edges = set()
        orange_edge_data = {}  # Store full edge data for drawing
        
        if closest_to_point1:
            edge_id, edge_data, shape_points, distance, angle_diff = closest_to_point1
            orange_edges.add(edge_id)
            orange_edge_data[edge_id] = (edge_data, shape_points, distance, angle_diff, "point 1")
            self.log(f"🟠 Closest edge to start point (point 1): {edge_id} (distance={distance:.1f}m, angle_diff={math.degrees(angle_diff):.1f}°)")
        
        if closest_to_point2:
            edge_id, edge_data, shape_points, distance, angle_diff = closest_to_point2
            orange_edges.add(edge_id)
            if edge_id in orange_edge_data:
                # Same edge for both points
                self.log(f"🟠 Closest edge to end point (point 2): {edge_id} (distance={distance:.1f}m, angle_diff={math.degrees(angle_diff):.1f}°) - SAME as point 1")
            else:
                orange_edge_data[edge_id] = (edge_data, shape_points, distance, angle_diff, "point 2")
                self.log(f"🟠 Closest edge to end point (point 2): {edge_id} (distance={distance:.1f}m, angle_diff={math.degrees(angle_diff):.1f}°)")
        
        if not orange_edges:
            self.log("⚠️ No edges found with matching direction for either point")
            return
        
        # Color edges (Y-flip for display)
        orange_pen = QPen(QColor(255, 165, 0), 5)  # Orange, 5px width (thicker for visibility)
        orange_pen.setStyle(Qt.SolidLine)
        
        # Draw orange edges (closest to point 1 and point 2)
        orange_count = 0
        for edge_id, (edge_data, shape_points, distance, angle_diff, point_label) in orange_edge_data.items():
            for i in range(len(shape_points) - 1):
                x1_sumo, y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL SUMO
                x2_sumo, y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL SUMO
                
                # Y-flip ONLY for display
                y1 = flip_y(y1_sumo)
                y2 = flip_y(y2_sumo)
                
                line = self.map_view.scene.addLine(x1_sumo, y1, x2_sumo, y2, orange_pen)
                line.setZValue(21)  # Above green edges (20) but below segment (400)
                self._green_edge_items.append(line)  # Store for clearing later
            
            self.log(f"  🟠 ORANGE Edge {edge_id} (closest to {point_label}): distance={distance:.1f}m, angle_diff={math.degrees(angle_diff):.1f}°")
            orange_count += 1
        
        # Draw green edges only if requested (top 5 segment matches, excluding orange ones)
        green_count = 0
        if show_green_edges:
            # Sort segment-matching edges by combined score for green edges display
            matching_edges.sort(key=lambda x: x[5])  # Sort by combined_score (index 5)
            top_5_edges = matching_edges[:5] if len(matching_edges) >= 5 else matching_edges
            
            # Filter out orange edges from green edges list
            green_edges = [edge for edge in top_5_edges if edge[0] not in orange_edges]
            
            green_pen = QPen(QColor(0, 255, 0), 4)  # Green, 4px width
            green_pen.setStyle(Qt.SolidLine)
            
            for edge_id, edge_data, shape_points, distance, angle_diff, combined_score in green_edges:
                for i in range(len(shape_points) - 1):
                    x1_sumo, y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL SUMO
                    x2_sumo, y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL SUMO
                    
                    # Y-flip ONLY for display
                    y1 = flip_y(y1_sumo)
                    y2 = flip_y(y2_sumo)
                    
                    line = self.map_view.scene.addLine(x1_sumo, y1, x2_sumo, y2, green_pen)
                    line.setZValue(20)  # Above red edges (15) but below orange (21)
                    self._green_edge_items.append(line)
                
                self.log(f"  🟢 green Edge {edge_id}: distance={distance:.1f}m, angle_diff={math.degrees(angle_diff):.1f}°, score={combined_score:.2f}")
                green_count += 1
        
        # Log summary
        if show_green_edges:
            self.log(f"🟢 Found {len(matching_edges)} edges with matching direction (threshold: {math.degrees(direction_threshold):.1f}°)")
            self.log(f"🟠 {len(orange_edges)} edge(s) marked as orange (closest to start/end points)")
            self.log(f"✓ Drew {orange_count} edge(s) in orange and {green_count} edge(s) in green")
        else:
            self.log(f"🟠 {len(orange_edges)} edge(s) marked as orange (closest to start/end points)")
            self.log(f"✓ Drew {orange_count} edge(s) in orange")

    def _run_route_finding(self) -> None:
        """Run route finding (all edges, same green/orange + Dijkstra/A* as view_network).
        Draws orange edges, green edges, and computed path(s) on the map.
        """
        if not self.network_parser or not self._current_sumo_segments:
            return

        # Clear previous orange/green/path overlay
        for item in self._green_edge_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                pass
        self._green_edge_items = []
        for item in self._route_path_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                pass
        self._route_path_items = []

        y_min = getattr(self.map_view, "_network_y_min", 0)
        y_max = getattr(self.map_view, "_network_y_max", 0)

        def flip_y(y: float) -> float:
            return y_max + y_min - y

        # All edges (same as view_network)
        edges_data = build_edges_data(self.network_parser)
        edge_shapes: Dict[str, List[List[float]]] = {
            eid: shape for eid, _ed, shape in edges_data
        }
        node_positions = build_node_positions(self.network_parser)

        orange_pen = QPen(QColor(255, 165, 0), 5)
        orange_pen.setStyle(Qt.SolidLine)
        green_pen = QPen(QColor(0, 255, 0), 4)
        green_pen.setStyle(Qt.SolidLine)
        path_pen = QPen(QColor(128, 0, 128), 5)  # Purple (same idea as view_network path)
        path_pen.setStyle(Qt.SolidLine)

        def draw_edge_lines(edge_ids: Set[str], pen: QPen, z_value: int, store_in: List) -> None:
            for eid in edge_ids:
                shape_points = edge_shapes.get(eid)
                if not shape_points or len(shape_points) < 2:
                    continue
                for i in range(len(shape_points) - 1):
                    x1, y1_sumo = shape_points[i][0], shape_points[i][1]
                    x2, y2_sumo = shape_points[i + 1][0], shape_points[i + 1][1]
                    y1, y2 = flip_y(y1_sumo), flip_y(y2_sumo)
                    line = self.map_view.scene.addLine(x1, y1, x2, y2, pen)
                    line.setZValue(z_value)
                    store_in.append(line)

        for seg_idx, seg_sumo in enumerate(self._current_sumo_segments):
            orange_ids, green_ids, start_id, end_id, candidates = compute_green_orange_edges(
                edges_data, seg_sumo, y_min, y_max, top_per_segment=5
            )
            if not start_id or not end_id:
                self.log(f"Segment {seg_idx + 1}: no start/end edge from trajectory")
                continue
            goal_xy = seg_sumo[-1]
            path_edges: List[str] = []
            max_tries = min(5, len(candidates or []))
            for try_idx in range(max_tries):
                cand = (candidates or [])[try_idx]
                path_edges = shortest_path_dijkstra(
                    self.network_parser,
                    cand,
                    end_id,
                    orange_ids=orange_ids,
                    green_ids=green_ids,
                    node_positions=node_positions,
                    goal_xy=goal_xy,
                )
                if path_edges:
                    if try_idx > 0:
                        self.log(
                            f"Segment {seg_idx + 1}: route found using start candidate {try_idx + 1}/{max_tries}"
                        )
                    break
            if path_edges:
                self.log(f"Segment {seg_idx + 1}: shortest path {len(path_edges)} edges")
                draw_edge_lines(
                    set(path_edges), path_pen, 25, self._route_path_items
                )
                # Build path polyline in display coords (same as seg_sumo) for projection
                path_points: List[Tuple[float, float]] = []
                for eid in path_edges:
                    shape_points = edge_shapes.get(eid)
                    if not shape_points:
                        continue
                    for x_s, y_s in shape_points:
                        path_points.append((x_s, flip_y(y_s)))
                # Draw stars on path: project each trajectory point onto path (like view_network)
                if path_points and seg_sumo:
                    from PySide6.QtGui import QPainterPath
                    from PySide6.QtWidgets import QGraphicsPathItem

                    def create_star_path(cx: float, cy: float, radius: float) -> QPainterPath:
                        path = QPainterPath()
                        num_points = 5
                        outer, inner = radius, radius * 0.4
                        for i in range(num_points * 2):
                            angle = (i * math.pi) / num_points - math.pi / 2
                            r = outer if i % 2 == 0 else inner
                            x = cx + r * math.cos(angle)
                            y = cy + r * math.sin(angle)
                            if i == 0:
                                path.moveTo(x, y)
                            else:
                                path.lineTo(x, y)
                        path.closeSubpath()
                        return path

                    star_radius = 7.2
                    star_brush = QBrush(QColor(255, 165, 0))
                    star_pen = QPen(QColor(255, 165, 0), 1.8)
                    font = QFont()
                    font.setPointSize(7)
                    font.setBold(True)
                    red_number_color = QColor(200, 0, 0)
                    path_points_flat = [[p[0], p[1]] for p in path_points]
                    for idx, (px, py) in enumerate(seg_sumo):
                        proj_x, proj_y = project_point_onto_polyline(px, py, path_points_flat)
                        star_path = create_star_path(proj_x, proj_y, star_radius)
                        star_item = QGraphicsPathItem(star_path)
                        star_item.setBrush(star_brush)
                        star_item.setPen(star_pen)
                        star_item.setZValue(30)
                        self.map_view.scene.addItem(star_item)
                        self._route_path_items.append(star_item)
                        text_str = str(idx + 1)
                        label_offset = star_radius
                        tx, ty = proj_x + label_offset, proj_y - label_offset
                        outline_offset = 0.5
                        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            outline_item = QGraphicsTextItem(text_str)
                            outline_item.setDefaultTextColor(QColor(0, 0, 0))
                            outline_item.setFont(font)
                            outline_item.setPos(tx + dx * outline_offset, ty + dy * outline_offset)
                            outline_item.setZValue(31)
                            self.map_view.scene.addItem(outline_item)
                            self._route_path_items.append(outline_item)
                        text_item = QGraphicsTextItem(text_str)
                        text_item.setDefaultTextColor(red_number_color)
                        text_item.setFont(font)
                        text_item.setPos(tx, ty)
                        text_item.setZValue(32)
                        self.map_view.scene.addItem(text_item)
                        self._route_path_items.append(text_item)
            else:
                self.log(
                    f"Segment {seg_idx + 1}: no route (tried up to {max_tries} start candidates)"
                )
            draw_edge_lines(orange_ids, orange_pen, 22, self._green_edge_items)
            draw_edge_lines(green_ids, green_pen, 21, self._green_edge_items)

        self.log("✓ Route finding complete (orange/green/path from all edges, Dijkstra/A*)")

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

            # Clear previous items before drawing new trajectory
            for item in self._route_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    pass
            self._route_items = []
            for item in self._green_edge_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    pass
            self._green_edge_items = []
            for item in self._route_path_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    pass
            self._route_path_items = []
            self._bounding_box_polygon = None

            # Split at invalid GPS jumps (same as view_network)
            segments = split_at_invalid_segments(polyline, DEFAULT_MAX_SEGMENT_DISTANCE)
            if len(segments) > 1:
                self.log(
                    f"Trajectory split into {len(segments)} segments "
                    f"(max {DEFAULT_MAX_SEGMENT_DISTANCE:.0f}m between points)"
                )
            # Trim each segment (same logic as view_network: 15m static threshold)
            segments_trimmed = []
            for seg in segments:
                trimmed = route_apply_trimming(seg)
                if len(trimmed) >= 2:
                    segments_trimmed.append(trimmed)
            if not segments_trimmed:
                self.log("❌ No valid segments after splitting and trimming")
                return

            # Convert each segment to SUMO (display coords) for route finding
            y_min = getattr(self.map_view, "_network_y_min", 0)
            y_max = getattr(self.map_view, "_network_y_max", 0)

            def flip_y(y):
                return y_max + y_min - y

            seg_sumos: List[List[Tuple[float, float]]] = []
            for seg in segments_trimmed:
                pts: List[Tuple[float, float]] = []
                for lon, lat in seg:
                    coords = self.network_parser.gps_to_sumo_coords(lon, lat)
                    if coords:
                        x, y = coords
                        pts.append((x, flip_y(y)))
                if len(pts) >= 2:
                    seg_sumos.append(pts)
            self._current_sumo_segments = seg_sumos

            # Concatenate polylines for display (one trajectory with all segments)
            all_polyline: List[List[float]] = []
            for seg in segments_trimmed:
                all_polyline.extend(seg)
            self.log(
                f"✓ After trimming: {len(all_polyline)} points in {len(seg_sumos)} segment(s)"
            )

            # Draw trajectory (use concatenated polyline; bounding box from original)
            self._draw_trajectory(all_polyline, original_polyline=polyline)
            self.log(f"✓ Trajectory {trajectory_num} drawn on map")

            # Run route finding (all edges, same green/orange + Dijkstra/A* as view_network)
            self._run_route_finding()
            
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
        for i in range(len(polyline) - 1):
            distance = haversine_distance(polyline[i], polyline[i + 1])
            # print the segment and distance
            self.log(f"  Segment {i+1} from {i+1} to {i+2}: distance={distance:.1f}m")

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
    
    def _draw_trajectory(self, polyline: List[List[float]], original_polyline: List[List[float]] = None):
        """Draw GPS points and connecting line on map.
        
        Args:
            polyline: Trimmed polyline to draw (for display)
            original_polyline: Optional original (untrimmed) polyline for bounding box calculation.
                              If None, uses polyline for both drawing and bounding box.
        """
        if not polyline or len(polyline) < 2:
            return
        
        # Use original polyline for bounding box if provided, otherwise use trimmed polyline
        bounding_box_polyline = original_polyline if original_polyline and len(original_polyline) >= 2 else polyline
        
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
        sumo_points_original = []  # Store in ORIGINAL SUMO coordinates (for trimmed polyline - display)
        for idx, (lon, lat) in enumerate(polyline):
            sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if sumo_coords:
                x, y = sumo_coords  # ORIGINAL SUMO coordinates
                sumo_points_original.append((x, y))
                
                # Debug: Log first 2 GPS points conversion
                if idx < 2:
                    self.log(f"  [GPS Conversion] Point {idx+1}: GPS=({lon:.6f}, {lat:.6f}) -> SUMO=({x:.1f}, {y:.1f}) [ORIGINAL, not Y-flipped]")
        
        # Convert bounding box polyline (original untrimmed) to SUMO coordinates for bounding box calculation
        sumo_points_original_for_bbox = []  # Store in ORIGINAL SUMO coordinates (for bounding box)
        if bounding_box_polyline != polyline:
            self.log(f"  Using original polyline ({len(bounding_box_polyline)} points) for bounding box calculation")
            failed_conversions = []
            for idx, (lon, lat) in enumerate(bounding_box_polyline):
                sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
                if sumo_coords:
                    x, y = sumo_coords  # ORIGINAL SUMO coordinates
                    sumo_points_original_for_bbox.append((x, y))
                    # Log first 2 and last 2 points for debugging
                    if idx < 2 or idx >= len(bounding_box_polyline) - 2:
                        self.log(f"  [BBox] Point {idx+1}/{len(bounding_box_polyline)}: GPS=({lon:.6f}, {lat:.6f}) -> SUMO=({x:.1f}, {y:.1f})")
                else:
                    failed_conversions.append(idx + 1)
                    self.log(f"  ⚠️ [BBox] Point {idx+1}/{len(bounding_box_polyline)}: GPS=({lon:.6f}, {lat:.6f}) -> FAILED to convert")
            if failed_conversions:
                self.log(f"  ⚠️ Warning: {len(failed_conversions)} points failed to convert (indices: {failed_conversions})")
            self.log(f"  Successfully converted {len(sumo_points_original_for_bbox)}/{len(bounding_box_polyline)} points for bounding box")
        else:
            # Use same points if no original polyline provided
            sumo_points_original_for_bbox = sumo_points_original
        
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
        # Use original (untrimmed) polyline points for bounding box to include all GPS points
        # Then Y-flip for display
        sumo_points_flipped_for_bbox = [(x, flip_y(y)) for x, y in sumo_points_original_for_bbox]
        self._draw_bounding_box(sumo_points_original_for_bbox, sumo_points_flipped_for_bbox)
        
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
            self.process_all_segments_btn.setEnabled(True)
            self.log(f"  Segment selection enabled: {max_segments} segments available")
        else:
            self.show_segment_btn.setEnabled(False)
            self.process_all_segments_btn.setEnabled(False)
        
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
        
        # CRITICAL: Ensure ALL points are inside the bounding box
        # Expand box if necessary to guarantee inclusion
        from PySide6.QtGui import QPolygonF
        polygon_check = QPolygonF(rotated_corners)
        outside_points_list = []
        for idx, (x, y) in enumerate(sumo_points_original):
            point = QPointF(x, y)
            if not polygon_check.containsPoint(point, Qt.OddEvenFill):
                outside_points_list.append((idx, x, y))
        
        # If ANY points are outside, expand the box to include them
        if outside_points_list:
            self.log(f"⚠️ {len(outside_points_list)} points are outside initial bounding box - expanding to include ALL points")
            # Log which points are outside (especially first 2)
            for idx, x, y in outside_points_list[:5]:
                self.log(f"  Outside point #{idx+1} (index {idx}): ({x:.1f}, {y:.1f})")
            
            # Expand box to guarantee all points are inside
            center_x, center_y, width, height, angle = self._expand_box_to_include_all_points(
                sumo_points_original, center_x, center_y, width, height, angle, safety_margin=100.0
            )
            
            # Recalculate corners with expanded box
            half_width = width / 2
            half_height = height / 2
            corners = [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height)
            ]
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            rotated_corners = []
            for dx, dy in corners:
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rotated_corners.append(QPointF(center_x + rx, center_y + ry))
            
            # Update stored bounding box parameters
            self._bounding_box_params = {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'angle': angle
            }
            self.log(f"📦 Expanded bounding box: center=({center_x:.1f}, {center_y:.1f}), size=({width:.1f}, {height:.1f}), angle={math.degrees(angle):.1f}°")
            
            # Verify again that all points are now inside
            polygon_check = QPolygonF(rotated_corners)
            still_outside = []
            for idx, (x, y) in enumerate(sumo_points_original):
                point = QPointF(x, y)
                if not polygon_check.containsPoint(point, Qt.OddEvenFill):
                    still_outside.append((idx, x, y))
            
            if still_outside:
                self.log(f"❌ ERROR: {len(still_outside)} points are STILL outside after expansion!")
                for idx, x, y in still_outside[:5]:
                    self.log(f"  Still outside point #{idx+1} (index {idx}): ({x:.1f}, {y:.1f})")
            else:
                self.log(f"✓ All {len(sumo_points_original)} points are now inside the expanded bounding box")
        
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
        """Color edges inside the bounding box polygon in red (all edge types)."""
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
        
        # Red pen for edges inside bounding box
        red_pen = QPen(QColor(255, 0, 0), 3)  # Red, 3px width
        red_pen.setStyle(Qt.SolidLine)
        
        edges_in_box = []
        edges_checked = 0
        
        for edge_id, edge_data in edges.items():
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
                # Store edge_id, full edge_data, and shape_points (ORIGINAL SUMO coordinates)
                edges_in_box.append((edge_id, edge_data, shape_points))
        
        # Store red edges data for later use in segment matching
        self._red_edges_data = edges_in_box.copy()
        # Enable export bounding box network when we have edges in box
        if getattr(self, 'export_bbox_network_btn', None) is not None:
            self.export_bbox_network_btn.setEnabled(len(edges_in_box) > 0)
        
        self.log(f"🔴 Found {len(edges_in_box)} edges inside bounding box (checked {edges_checked} edges)")
        
        # Draw edges in red (Y-flip for display)
        for edge_id, edge_data, shape_points in edges_in_box:
            for i in range(len(shape_points) - 1):
                x1, y1_sumo = shape_points[i][0], shape_points[i][1]  # ORIGINAL SUMO
                x2, y2_sumo = shape_points[i+1][0], shape_points[i+1][1]  # ORIGINAL SUMO
                
                # Y-flip ONLY for display
                y1 = flip_y(y1_sumo)
                y2 = flip_y(y2_sumo)
                
                line = self.map_view.scene.addLine(x1, y1, x2, y2, red_pen)
                line.setZValue(15)  # Above network edges (0) but below other overlays
                self._route_items.append(line)
        
        self.log(f"✓ Drew {len(edges_in_box)} edges in red inside bounding box")
    
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
    
    def _expand_box_to_include_all_points(self, points: List[Tuple[float, float]], center_x: float, center_y: float, 
                                         width: float, height: float, angle: float, 
                                         safety_margin: float = 50.0) -> Tuple[float, float, float, float, float]:
        """
        Expand bounding box to guarantee all points are inside.
        
        Args:
            points: List of (x, y) tuples
            center_x, center_y, width, height, angle: Current bounding box parameters
            safety_margin: Additional margin to add (in SUMO coordinate units)
        
        Returns:
            Updated (center_x, center_y, width, height, angle) that includes all points
        """
        if not points:
            return (center_x, center_y, width, height, angle)
        
        # Transform all points to the rotated coordinate system (aligned with box)
        cos_neg_angle = math.cos(-angle)
        sin_neg_angle = math.sin(-angle)
        
        # Transform points to rotated coordinate system
        rotated_points = []
        for x, y in points:
            # Translate to center
            dx = x - center_x
            dy = y - center_y
            # Rotate by -angle (to align with box axes)
            rx = dx * cos_neg_angle - dy * sin_neg_angle
            ry = dx * sin_neg_angle + dy * cos_neg_angle
            rotated_points.append((rx, ry))
        
        # Find the axis-aligned bounding box of rotated points
        if not rotated_points:
            return (center_x, center_y, width, height, angle)
        
        rxs = [p[0] for p in rotated_points]
        rys = [p[1] for p in rotated_points]
        
        min_rx = min(rxs)
        max_rx = max(rxs)
        min_ry = min(rys)
        max_ry = max(rys)
        
        # Calculate required width and height (with safety margin)
        required_width = (max_rx - min_rx) + 2 * safety_margin
        required_height = (max_ry - min_ry) + 2 * safety_margin
        
        # Calculate new center in rotated coordinates (centered on the points)
        center_rx = (min_rx + max_rx) / 2
        center_ry = (min_ry + max_ry) / 2
        
        # Transform center back to original coordinates
        # We rotated by -angle, so to go back we rotate by +angle
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        new_center_x = center_x + center_rx * cos_angle - center_ry * sin_angle
        new_center_y = center_y + center_rx * sin_angle + center_ry * cos_angle
        
        # Use the larger of current or required dimensions
        final_width = max(width, required_width)
        final_height = max(height, required_height)
        
        return (new_center_x, new_center_y, final_width, final_height, angle)
    
    def _verify_points_in_box(self, points: List[Tuple[float, float]], box_corners: List[QPointF]):
        """Verify that all points are inside the bounding box."""
        if len(box_corners) < 4:
            return
        
        # Check if points are inside using point-in-polygon test
        outside_points = []
        outside_indices = []
        for idx, (x, y) in enumerate(points):
            point = QPointF(x, y)
            # Use Qt's built-in point-in-polygon test
            from PySide6.QtGui import QPolygonF
            polygon = QPolygonF(box_corners)
            inside = polygon.containsPoint(point, Qt.OddEvenFill)
            
            if not inside:
                outside_points.append((x, y))
                outside_indices.append(idx)
        
        if outside_points:
            self.log(f"⚠️ Warning: {len(outside_points)} points are outside the bounding box!")
            # Log first 5 and last 5 outside points with their indices
            for i, (idx, (x, y)) in enumerate(zip(outside_indices[:5], outside_points[:5])):
                self.log(f"  Outside point #{idx+1} (index {idx}): ({x:.1f}, {y:.1f})")
            if len(outside_points) > 5:
                self.log(f"  ... and {len(outside_points) - 5} more outside points")
                for idx, (x, y) in zip(outside_indices[-5:], outside_points[-5:]):
                    self.log(f"  Outside point #{idx+1} (index {idx}): ({x:.1f}, {y:.1f})")
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
