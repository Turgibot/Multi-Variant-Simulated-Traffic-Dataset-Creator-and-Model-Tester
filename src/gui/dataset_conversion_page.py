"""
Dataset Conversion page for Porto taxi data.
Provides map visualization and dataset conversion controls.
"""

import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Tuple

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea,
                               QStackedWidget, QTextEdit, QVBoxLayout, QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.trip_validator import (TripValidationResult,
                                      validate_trip_segments)

# Default SUMO_HOME path
DEFAULT_SUMO_HOME = "/usr/share/sumo"

# Settings file name
SETTINGS_FILE = "porto_settings.json"

# Porto bounding box coordinates (wider area to cover taxi routes)
PORTO_BBOX = {
    'north': 41.22,
    'south': 41.10,
    'east': -8.30,
    'west': -8.80
}


class NetworkLoaderWorker(QThread):
    """Worker thread for loading network files asynchronously."""
    
    progress = Signal(int)  # Progress percentage
    status = Signal(str)  # Status message
    finished = Signal(bool, object, str)  # Success, NetworkParser or None, message
    
    def __init__(self, net_file: str, parent=None):
        super().__init__(parent)
        self.net_file = net_file
    
    def run(self):
        """Load the network file."""
        try:
            self.status.emit("Loading network file...")
            self.progress.emit(20)
            
            # Parse the network
            self.status.emit("Parsing network geometry...")
            self.progress.emit(50)
            
            network_parser = NetworkParser(self.net_file)
            
            self.progress.emit(80)
            self.status.emit("Preparing map view...")
            
            # Get statistics
            edges = network_parser.get_edges()
            nodes = network_parser.get_nodes()
            
            self.progress.emit(100)
            self.finished.emit(True, network_parser, f"Loaded {len(edges)} edges, {len(nodes)} nodes")
            
        except Exception as e:
            self.finished.emit(False, None, str(e))


class NetworkFilterWorker(QThread):
    """Worker thread for preparing filtered network data."""
    
    finished = Signal(bool, bool, dict)  # Success, filter_enabled, stats dict
    
    def __init__(self, network_parser, roads_junctions_only: bool, parent=None):
        super().__init__(parent)
        self.network_parser = network_parser
        self.roads_junctions_only = roads_junctions_only
    
    def run(self):
        """Prepare filtered network statistics."""
        try:
            edges = self.network_parser.get_edges()
            
            if self.roads_junctions_only:
                # Count filtered items
                filtered_edges = sum(1 for e in edges.values() if e.get('allows_passenger', True))
                junctions = self.network_parser.get_junctions()
                stats = {
                    'edge_count': filtered_edges,
                    'node_count': len(junctions),
                    'filtered': True
                }
            else:
                nodes = self.network_parser.get_nodes()
                stats = {
                    'edge_count': len(edges),
                    'node_count': len(nodes),
                    'filtered': False
                }
            
            self.finished.emit(True, self.roads_junctions_only, stats)
        except Exception as e:
            self.finished.emit(False, self.roads_junctions_only, {'error': str(e)})


class DownloadWorker(QThread):
    """Worker thread for downloading files."""
    
    progress = Signal(int)  # Progress percentage
    status = Signal(str)  # Status message
    finished = Signal(bool, str)  # Success, message
    
    def __init__(self, task_type: str, output_path: str, sumo_home: str = None, parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.output_path = output_path
        self.sumo_home = sumo_home or DEFAULT_SUMO_HOME
        self._is_cancelled = False
    
    def cancel(self):
        """Cancel the download."""
        self._is_cancelled = True
    
    def run(self):
        """Run the download task."""
        try:
            if self.task_type == "map":
                self._download_map()
            else:
                self.finished.emit(False, f"Unknown task type: {self.task_type}")
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def _download_map(self):
        """Download Porto OSM data and convert to SUMO network."""
        output_dir = Path(self.output_path)
        osm_file = output_dir / 'porto.osm'
        net_file = output_dir / 'porto.net.xml'
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Download OSM data (0-60%)
        self.status.emit("Downloading Porto OSM data from Overpass API...")
        self.progress.emit(5)
        
        if self._is_cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        query = f"""
        [out:xml][timeout:120];
        (
          way["highway"]({PORTO_BBOX['south']},{PORTO_BBOX['west']},{PORTO_BBOX['north']},{PORTO_BBOX['east']});
          relation["highway"]({PORTO_BBOX['south']},{PORTO_BBOX['west']},{PORTO_BBOX['north']},{PORTO_BBOX['east']});
        );
        (._;>;);
        out body;
        """
        
        url = "https://overpass-api.de/api/interpreter"
        
        try:
            req = urllib.request.Request(url, data=query.encode('utf-8'))
            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = response.headers.get('Content-Length')
                if total_size:
                    total_size = int(total_size)
                
                downloaded = 0
                chunk_size = 8192
                with open(osm_file, 'wb') as f:
                    while True:
                        if self._is_cancelled:
                            self.finished.emit(False, "Cancelled")
                            return
                        
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size:
                            progress = int(5 + (downloaded / total_size) * 55)
                            self.progress.emit(min(progress, 60))
                        else:
                            # If no content length, estimate progress
                            self.progress.emit(min(30, 5 + downloaded // 100000))
            
            self.progress.emit(60)
            self.status.emit("OSM data downloaded successfully")
            
        except Exception as e:
            self.finished.emit(False, f"Failed to download OSM data: {e}")
            return
        
        # Step 2: Convert to SUMO network (60-100%)
        self.status.emit("Converting to SUMO network format...")
        self.progress.emit(65)
        
        if self._is_cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        # Find netconvert using provided SUMO_HOME
        netconvert_path = None
        
        # First try the provided SUMO_HOME path
        if self.sumo_home:
            netconvert_path = Path(self.sumo_home) / 'bin' / 'netconvert'
            if not netconvert_path.exists():
                netconvert_path = None
                self.status.emit(f"netconvert not found at {self.sumo_home}/bin/")
        
        # Try environment variable as fallback
        if not netconvert_path:
            env_sumo_home = os.environ.get('SUMO_HOME')
            if env_sumo_home:
                netconvert_path = Path(env_sumo_home) / 'bin' / 'netconvert'
                if not netconvert_path.exists():
                    netconvert_path = None
        
        # Try to find in PATH as last resort
        if not netconvert_path:
            try:
                result = subprocess.run(['which', 'netconvert'], capture_output=True, text=True)
                if result.returncode == 0:
                    netconvert_path = Path(result.stdout.strip())
            except Exception:
                pass
        
        if not netconvert_path or not netconvert_path.exists():
            self.finished.emit(False, f"SUMO netconvert not found.\n\nChecked paths:\n• {self.sumo_home}/bin/netconvert\n• System PATH\n\nPlease verify SUMO_HOME path is correct.")
            return
        
        self.progress.emit(70)
        
        # Build netconvert command
        typemap_file = Path(self.sumo_home) / 'data' / 'typemap' / 'osmNetconvert.typ.xml'
        
        cmd = [
            str(netconvert_path),
            '--osm-files', str(osm_file),
            '--output-file', str(net_file),
            '--geometry.remove',
            '--roundabouts.guess',
            '--ramps.guess',
            '--junctions.join',
            '--tls.guess-signals',
            '--tls.discard-simple',
            '--tls.join',
            '--no-turnarounds',
            '--no-internal-links',
        ]
        
        # Add typemap file if it exists
        if typemap_file.exists():
            cmd.extend(['--type-files', str(typemap_file)])
        
        try:
            self.status.emit("Running netconvert (this may take a while)...")
            
            # Set up environment with SUMO_HOME
            env = os.environ.copy()
            env['SUMO_HOME'] = self.sumo_home
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            
            if result.returncode == 0:
                self.progress.emit(95)
                
                # Also create route file and sumocfg
                self._create_route_and_config(output_dir)
                
                self.progress.emit(100)
                self.finished.emit(True, "Porto network downloaded and converted successfully!")
            else:
                self.finished.emit(False, f"Conversion failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Conversion timed out (took more than 10 minutes)")
        except Exception as e:
            self.finished.emit(False, f"Conversion error: {e}")
    
    def _create_route_and_config(self, output_dir: Path):
        """Create route file and SUMO config."""
        route_file = output_dir / 'porto.rou.xml'
        sumocfg_file = output_dir / 'porto.sumocfg'
        
        # Create empty route file
        if not route_file.exists():
            route_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
</routes>
'''
            with open(route_file, 'w', encoding='utf-8') as f:
                f.write(route_xml)
        
        # Create sumocfg
        if not sumocfg_file.exists():
            config_xml = '''<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="porto.net.xml"/>
        <route-files value="porto.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <lateral-resolution value="0.8"/>
    </processing>
    
    <report>
        <verbose value="true"/>
        <no-warnings value="false"/>
    </report>
</configuration>
'''
            with open(sumocfg_file, 'w', encoding='utf-8') as f:
                f.write(config_xml)


class DatasetConversionPage(QWidget):
    """Page for Porto dataset conversion with map view."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.network_parser = None
        self.download_worker = None
        self.network_loader_worker = None
        self._loading_settings = False  # Flag to prevent save during load
        self._train_trip_count = None  # Cached trip count
        self._route_items = []  # Graphics items for current route display
        
        self.init_ui()
        
        # Load saved settings
        self.load_settings()
        
        # Use QTimer to check resources after UI is shown (non-blocking)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self.check_resources)
    
    def init_ui(self):
        """Initialize the UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
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
        
        title = QLabel(f"Dataset Conversion - {self.project_name}")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #FF9800;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Main content area - 70% map, 30% controls
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # ========== LEFT: Map View (70%) ==========
        map_container = QFrame()
        map_container.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 2px solid #ddd;
                border-radius: 8px;
            }
        """)
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(10, 10, 10, 10)
        
        # Map header with title and zoom controls
        map_header_layout = QHBoxLayout()
        map_header_layout.setContentsMargins(0, 0, 0, 5)
        
        # Map title
        map_title = QLabel("🗺️ Porto Network Map")
        map_title_font = QFont()
        map_title_font.setPointSize(14)
        map_title_font.setBold(True)
        map_title.setFont(map_title_font)
        map_title.setStyleSheet("color: #333;")
        map_header_layout.addWidget(map_title)
        
        map_header_layout.addSpacing(20)
        
        # Roads and junctions only checkbox
        self.roads_junctions_only_checkbox = QCheckBox("Roads and junctions only")
        self.roads_junctions_only_checkbox.setToolTip(
            "When checked, only show road edges and junction nodes\n"
            "(hides pedestrian paths, rail tracks, and other non-road elements)"
        )
        self.roads_junctions_only_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #999;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4CAF50;
                background-color: white;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                background-color: white;
            }
        """)
        self.roads_junctions_only_checkbox.stateChanged.connect(self.on_roads_junctions_filter_changed)
        self.roads_junctions_only_checkbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.roads_junctions_only_checkbox)
        
        # Loading indicator for filter
        self.filter_loading_label = QLabel("")
        self.filter_loading_label.setStyleSheet("font-size: 14px;")
        self.filter_loading_label.setFixedWidth(20)
        self.filter_loading_label.setVisible(False)
        map_header_layout.addWidget(self.filter_loading_label)
        
        map_header_layout.addSpacing(10)
        
        # Satellite imagery checkbox
        self.satellite_checkbox = QCheckBox("Show satellite imagery")
        self.satellite_checkbox.setToolTip(
            "When checked, display satellite imagery as map background\n"
            "(downloads tiles from ESRI World Imagery service)"
        )
        self.satellite_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #999;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #2196F3;
                background-color: white;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                background-color: white;
            }
        """)
        self.satellite_checkbox.stateChanged.connect(self.on_satellite_changed)
        self.satellite_checkbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.satellite_checkbox)
        
        # Loading indicator for satellite
        self.satellite_loading_label = QLabel("")
        self.satellite_loading_label.setStyleSheet("font-size: 14px;")
        self.satellite_loading_label.setFixedWidth(20)
        self.satellite_loading_label.setVisible(False)
        map_header_layout.addWidget(self.satellite_loading_label)
        
        map_header_layout.addStretch()
        
        # Zoom controls in header
        zoom_btn_style = """
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                padding: 4px 10px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
                min-width: 28px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """
        
        # Zoom In button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.setStyleSheet(zoom_btn_style)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        map_header_layout.addWidget(self.zoom_in_btn)
        
        # Zoom Out button
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.setStyleSheet(zoom_btn_style)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        map_header_layout.addWidget(self.zoom_out_btn)
        
        map_header_layout.addSpacing(8)
        
        # Default zoom button (Porto city center)
        self.zoom_default_btn = QPushButton("🏙️")
        self.zoom_default_btn.setToolTip("Zoom to Porto city center")
        self.zoom_default_btn.setStyleSheet(zoom_btn_style)
        self.zoom_default_btn.clicked.connect(self.zoom_to_default)
        map_header_layout.addWidget(self.zoom_default_btn)
        
        # Reset zoom button (show entire map)
        self.zoom_reset_btn = QPushButton("🗺️")
        self.zoom_reset_btn.setToolTip("Show entire map")
        self.zoom_reset_btn.setStyleSheet(zoom_btn_style)
        self.zoom_reset_btn.clicked.connect(self.zoom_to_full)
        map_header_layout.addWidget(self.zoom_reset_btn)
        
        map_layout.addLayout(map_header_layout)
        
        # Map view container (stacked widget for overlay)
        self.map_stack = QStackedWidget()
        self.map_stack.setMinimumSize(400, 400)
        
        # Page 0: Map view
        self.map_view = SimulationView()
        self.map_view.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #e8e8e8;
            }
        """)
        self.map_stack.addWidget(self.map_view)
        
        # Page 1: Loading overlay
        loading_widget = QWidget()
        loading_widget.setStyleSheet("background-color: #f0f0f0;")
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        
        # Loading card
        loading_card = QFrame()
        loading_card.setFixedSize(300, 150)
        loading_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #FF9800;
                border-radius: 12px;
            }
        """)
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        loading_card.setGraphicsEffect(shadow)
        
        loading_card_layout = QVBoxLayout(loading_card)
        loading_card_layout.setAlignment(Qt.AlignCenter)
        loading_card_layout.setSpacing(15)
        
        # Loading icon/text
        self.loading_label = QLabel("🗺️ Loading Map...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
        """)
        loading_card_layout.addWidget(self.loading_label)
        
        # Loading progress bar
        self.map_loading_progress = QProgressBar()
        self.map_loading_progress.setMinimum(0)
        self.map_loading_progress.setMaximum(100)
        self.map_loading_progress.setValue(0)
        self.map_loading_progress.setFixedWidth(250)
        self.map_loading_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 8px;
                text-align: center;
                background-color: #f5f5f5;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
                border-radius: 7px;
            }
        """)
        loading_card_layout.addWidget(self.map_loading_progress, alignment=Qt.AlignCenter)
        
        # Loading status
        self.loading_status_label = QLabel("Initializing...")
        self.loading_status_label.setAlignment(Qt.AlignCenter)
        self.loading_status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #666;
            }
        """)
        loading_card_layout.addWidget(self.loading_status_label)
        
        loading_layout.addWidget(loading_card)
        self.map_stack.addWidget(loading_widget)
        
        # Page 2: Busy overlay (for zoom operations)
        busy_widget = QWidget()
        busy_widget.setStyleSheet("background-color: rgba(240, 240, 240, 200);")
        busy_layout = QVBoxLayout(busy_widget)
        busy_layout.setAlignment(Qt.AlignCenter)
        
        # Busy indicator card
        busy_card = QFrame()
        busy_card.setFixedSize(200, 80)
        busy_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #607D8B;
                border-radius: 10px;
            }
        """)
        busy_shadow = QGraphicsDropShadowEffect()
        busy_shadow.setBlurRadius(15)
        busy_shadow.setOffset(0, 3)
        busy_card.setGraphicsEffect(busy_shadow)
        
        busy_card_layout = QVBoxLayout(busy_card)
        busy_card_layout.setAlignment(Qt.AlignCenter)
        busy_card_layout.setSpacing(8)
        
        self.busy_label = QLabel("⏳ Processing...")
        self.busy_label.setAlignment(Qt.AlignCenter)
        self.busy_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #455A64;
            }
        """)
        busy_card_layout.addWidget(self.busy_label)
        
        busy_layout.addWidget(busy_card)
        self.map_stack.addWidget(busy_widget)
        
        # Start with map view (page 0)
        self.map_stack.setCurrentIndex(0)
        
        map_layout.addWidget(self.map_stack, stretch=1)
        
        # Map status label
        self.map_status_label = QLabel("No map loaded")
        self.map_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        self.map_status_label.setAlignment(Qt.AlignCenter)
        map_layout.addWidget(self.map_status_label)
        
        content_layout.addWidget(map_container, stretch=70)
        
        # ========== RIGHT: Controls Panel (30%) with Scroll ==========
        # Create scroll area for controls
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #FF9800;
                border-radius: 8px;
                background-color: #fafafa;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #fafafa;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #FF9800;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #F57C00;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        controls_container = QWidget()
        controls_container.setStyleSheet("background-color: #fafafa;")
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        controls_layout.setSpacing(15)
        
        # Panel title
        panel_title = QLabel("🚕 Porto Dataset Controls")
        panel_title_font = QFont()
        panel_title_font.setPointSize(14)
        panel_title_font.setBold(True)
        panel_title.setFont(panel_title_font)
        panel_title.setStyleSheet("color: #FF9800; margin-bottom: 10px;")
        controls_layout.addWidget(panel_title)
        
        # ---- SUMO_HOME Section ----
        sumo_group = QGroupBox("🔧 SUMO Configuration")
        sumo_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #333;
            }
        """)
        sumo_group_layout = QVBoxLayout()
        sumo_group_layout.setSpacing(8)
        
        sumo_label = QLabel("SUMO_HOME Path:")
        sumo_label.setStyleSheet("color: #555; font-size: 11px; font-weight: normal;")
        sumo_group_layout.addWidget(sumo_label)
        
        # SUMO_HOME path input with browse and reset buttons
        sumo_path_layout = QHBoxLayout()
        sumo_path_layout.setSpacing(5)
        
        self.sumo_home_input = QLineEdit()
        self.sumo_home_input.setText(DEFAULT_SUMO_HOME)
        self.sumo_home_input.setPlaceholderText("e.g., /usr/share/sumo")
        self.sumo_home_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #fafafa;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #FF9800;
            }
        """)
        self.sumo_home_input.textChanged.connect(self.on_sumo_home_changed)
        sumo_path_layout.addWidget(self.sumo_home_input, stretch=1)
        
        # Browse button
        browse_btn = QPushButton("📁")
        browse_btn.setToolTip("Browse for SUMO installation folder")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        browse_btn.clicked.connect(self.browse_sumo_home)
        sumo_path_layout.addWidget(browse_btn)
        
        # Reset button
        reset_btn = QPushButton("↺")
        reset_btn.setToolTip("Reset to default path (/usr/share/sumo)")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        reset_btn.clicked.connect(self.reset_sumo_home)
        sumo_path_layout.addWidget(reset_btn)
        
        sumo_group_layout.addLayout(sumo_path_layout)
        
        # SUMO status label
        self.sumo_status_label = QLabel("")
        self.sumo_status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.sumo_status_label.setWordWrap(True)
        sumo_group_layout.addWidget(self.sumo_status_label)
        
        sumo_group.setLayout(sumo_group_layout)
        controls_layout.addWidget(sumo_group)
        
        # ---- Map Download Section ----
        map_group = QGroupBox("📍 Network Map")
        map_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #333;
            }
        """)
        map_group_layout = QVBoxLayout()
        map_group_layout.setSpacing(10)
        
        self.map_status = QLabel("Checking...")
        self.map_status.setWordWrap(True)
        self.map_status.setStyleSheet("color: #666; font-size: 11px;")
        map_group_layout.addWidget(self.map_status)
        
        self.download_map_btn = QPushButton("⬇️ Download & Render Map")
        self.download_map_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.download_map_btn.clicked.connect(self.download_map)
        map_group_layout.addWidget(self.download_map_btn)
        
        
        map_group.setLayout(map_group_layout)
        controls_layout.addWidget(map_group)
        
        # ---- Dataset Path Section ----
        dataset_group = QGroupBox("📊 Taxi Dataset")
        dataset_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #333;
            }
        """)
        dataset_group_layout = QVBoxLayout()
        dataset_group_layout.setSpacing(8)
        
        # Header row with label and help button
        dataset_header_layout = QHBoxLayout()
        dataset_label = QLabel("Dataset CSV Path:")
        dataset_label.setStyleSheet("color: #555; font-size: 11px; font-weight: normal;")
        dataset_header_layout.addWidget(dataset_label)
        dataset_header_layout.addStretch()
        
        # Help button
        help_btn = QPushButton("❓")
        help_btn.setToolTip("How to download the Porto taxi dataset")
        help_btn.setFixedSize(28, 28)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        help_btn.clicked.connect(self.show_dataset_help)
        dataset_header_layout.addWidget(help_btn)
        dataset_group_layout.addLayout(dataset_header_layout)
        
        # Common style for path inputs
        path_input_style = """
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #fafafa;
                font-size: 10px;
            }
            QLineEdit:focus {
                border-color: #FF9800;
            }
        """
        browse_btn_style = """
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """
        
        # ---- Train.csv path ----
        train_label = QLabel("Train CSV (train.csv):")
        train_label.setStyleSheet("color: #555; font-size: 10px; font-weight: normal; margin-top: 5px;")
        dataset_group_layout.addWidget(train_label)
        
        train_path_layout = QHBoxLayout()
        train_path_layout.setSpacing(4)
        
        self.train_path_input = QLineEdit()
        self.train_path_input.setPlaceholderText("Select train.csv...")
        self.train_path_input.setStyleSheet(path_input_style)
        self.train_path_input.textChanged.connect(lambda p: self.on_train_path_changed(p))
        train_path_layout.addWidget(self.train_path_input, stretch=1)
        
        browse_train_btn = QPushButton("📁")
        browse_train_btn.setToolTip("Browse for train.csv")
        browse_train_btn.setStyleSheet(browse_btn_style)
        browse_train_btn.clicked.connect(lambda: self.browse_dataset_file("train"))
        train_path_layout.addWidget(browse_train_btn)
        
        self.train_valid_label = QLabel("")
        self.train_valid_label.setFixedWidth(20)
        self.train_valid_label.setStyleSheet("font-size: 14px;")
        train_path_layout.addWidget(self.train_valid_label)
        
        dataset_group_layout.addLayout(train_path_layout)
        
        # Train status
        self.train_status = QLabel("")
        self.train_status.setStyleSheet("color: #666; font-size: 9px; margin-bottom: 5px;")
        dataset_group_layout.addWidget(self.train_status)
        
        # ---- Test.csv path ----
        test_label = QLabel("Test CSV (test.csv):")
        test_label.setStyleSheet("color: #555; font-size: 10px; font-weight: normal; margin-top: 3px;")
        dataset_group_layout.addWidget(test_label)
        
        test_path_layout = QHBoxLayout()
        test_path_layout.setSpacing(4)
        
        self.test_path_input = QLineEdit()
        self.test_path_input.setPlaceholderText("Select test.csv...")
        self.test_path_input.setStyleSheet(path_input_style)
        self.test_path_input.textChanged.connect(lambda p: self.on_test_path_changed(p))
        test_path_layout.addWidget(self.test_path_input, stretch=1)
        
        browse_test_btn = QPushButton("📁")
        browse_test_btn.setToolTip("Browse for test.csv")
        browse_test_btn.setStyleSheet(browse_btn_style)
        browse_test_btn.clicked.connect(lambda: self.browse_dataset_file("test"))
        test_path_layout.addWidget(browse_test_btn)
        
        self.test_valid_label = QLabel("")
        self.test_valid_label.setFixedWidth(20)
        self.test_valid_label.setStyleSheet("font-size: 14px;")
        test_path_layout.addWidget(self.test_valid_label)
        
        dataset_group_layout.addLayout(test_path_layout)
        
        # Test status
        self.test_status = QLabel("")
        self.test_status.setStyleSheet("color: #666; font-size: 9px;")
        dataset_group_layout.addWidget(self.test_status)
        
        dataset_group.setLayout(dataset_group_layout)
        controls_layout.addWidget(dataset_group)
        
        # ---- Number of Zones Section (hidden until map and dataset ready) ----
        self.zones_group = QGroupBox("🗂️ Zone Configuration")
        self.zones_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #2E7D32;
            }
        """)
        zones_group_layout = QVBoxLayout()
        zones_group_layout.setSpacing(8)
        
        # Zones label with value
        zones_header = QHBoxLayout()
        zones_label = QLabel("Number of Zones:")
        zones_label.setStyleSheet("color: #333; font-size: 11px; font-weight: normal;")
        zones_header.addWidget(zones_label)
        zones_header.addStretch()
        
        self.zones_value_label = QLabel("10")
        self.zones_value_label.setStyleSheet("""
            QLabel {
                color: #2E7D32;
                font-size: 14px;
                font-weight: bold;
                background-color: white;
                padding: 2px 10px;
                border-radius: 3px;
                border: 1px solid #4CAF50;
            }
        """)
        zones_header.addWidget(self.zones_value_label)
        zones_group_layout.addLayout(zones_header)
        
        # Zones slider
        from PySide6.QtWidgets import QSlider
        self.zones_slider = QSlider(Qt.Horizontal)
        self.zones_slider.setMinimum(1)
        self.zones_slider.setMaximum(50)
        self.zones_slider.setValue(10)
        self.zones_slider.setTickPosition(QSlider.TicksBelow)
        self.zones_slider.setTickInterval(5)
        self.zones_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #388E3C;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #388E3C;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)
        self.zones_slider.valueChanged.connect(self.on_zones_changed)
        zones_group_layout.addWidget(self.zones_slider)
        
        # Min/Max labels
        minmax_layout = QHBoxLayout()
        min_label = QLabel("1")
        min_label.setStyleSheet("color: #666; font-size: 9px;")
        minmax_layout.addWidget(min_label)
        minmax_layout.addStretch()
        max_label = QLabel("50")
        max_label.setStyleSheet("color: #666; font-size: 9px;")
        minmax_layout.addWidget(max_label)
        zones_group_layout.addLayout(minmax_layout)
        
        self.zones_group.setLayout(zones_group_layout)
        self.zones_group.setVisible(False)  # Hidden until map and dataset ready
        controls_layout.addWidget(self.zones_group)
        
        # ---- Route Display Section (hidden until map and dataset ready) ----
        self.route_group = QGroupBox("🛣️ Route Display")
        self.route_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #e3f2fd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #1565C0;
            }
        """)
        route_group_layout = QVBoxLayout()
        route_group_layout.setSpacing(8)
        
        # Number of trips display
        trips_layout = QHBoxLayout()
        trips_label = QLabel("Total Trips in Dataset:")
        trips_label.setStyleSheet("color: #333; font-size: 11px; font-weight: normal;")
        trips_layout.addWidget(trips_label)
        trips_layout.addStretch()
        
        self.trips_count_label = QLabel("Loading...")
        self.trips_count_label.setStyleSheet("""
            QLabel {
                color: #1565C0;
                font-size: 12px;
                font-weight: bold;
                background-color: white;
                padding: 2px 8px;
                border-radius: 3px;
                border: 1px solid #2196F3;
            }
        """)
        trips_layout.addWidget(self.trips_count_label)
        route_group_layout.addLayout(trips_layout)
        
        # Route number input
        route_input_label = QLabel("Display Route #:")
        route_input_label.setStyleSheet("color: #333; font-size: 11px; font-weight: normal;")
        route_group_layout.addWidget(route_input_label)
        
        route_input_layout = QHBoxLayout()
        route_input_layout.setSpacing(4)
        
        from PySide6.QtWidgets import QSpinBox
        self.route_spinbox = QSpinBox()
        self.route_spinbox.setMinimum(1)
        self.route_spinbox.setMaximum(1)  # Will be updated when dataset loads
        self.route_spinbox.setValue(1)
        self.route_spinbox.setStyleSheet("""
            QSpinBox {
                font-size: 12px;
                padding: 4px 8px;
                border: 1px solid #2196F3;
                border-radius: 4px;
                background-color: white;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
            }
        """)
        route_input_layout.addWidget(self.route_spinbox, stretch=1)
        
        self.show_route_btn = QPushButton("🗺️ Show")
        self.show_route_btn.setToolTip("Display selected route on map")
        self.show_route_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        self.show_route_btn.clicked.connect(self.show_selected_route)
        route_input_layout.addWidget(self.show_route_btn)
        
        self.clear_route_btn = QPushButton("🗑️ Clear")
        self.clear_route_btn.setToolTip("Clear route from map")
        self.clear_route_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.clear_route_btn.clicked.connect(self.clear_route_display)
        route_input_layout.addWidget(self.clear_route_btn)
        
        route_group_layout.addLayout(route_input_layout)
        
        # Route info label
        self.route_info_label = QLabel("")
        self.route_info_label.setStyleSheet("color: #666; font-size: 9px;")
        self.route_info_label.setWordWrap(True)
        route_group_layout.addWidget(self.route_info_label)
        
        # ---- Route Repair Subsection ----
        repair_layout = QHBoxLayout()
        repair_layout.setSpacing(8)
        
        # Trim Start/End checkbox (on the left)
        self.fix_route_checkbox = QCheckBox("Trim Start/End")
        self.fix_route_checkbox.setToolTip("Use real start and destination points instead of original pickup/dropoff points")
        self.fix_route_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.fix_route_checkbox.stateChanged.connect(self._on_fix_route_changed)
        repair_layout.addWidget(self.fix_route_checkbox)
        
        # Fix Invalid Segments checkbox
        self.fix_invalid_segments_checkbox = QCheckBox("Fix Invalid Segments")
        self.fix_invalid_segments_checkbox.setToolTip("Split route at invalid segments (>1000m) and treat each part as a separate trip")
        self.fix_invalid_segments_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.fix_invalid_segments_checkbox.stateChanged.connect(self._on_fix_route_changed)
        repair_layout.addWidget(self.fix_invalid_segments_checkbox)
        
        # Real start point label
        self.real_start_label = QLabel("Real Start: N/A")
        self.real_start_label.setStyleSheet("color: #333; font-size: 10px;")
        repair_layout.addWidget(self.real_start_label)
        
        # Real destination point label
        self.real_destination_label = QLabel("Real Destination: N/A")
        self.real_destination_label.setStyleSheet("color: #333; font-size: 10px;")
        repair_layout.addWidget(self.real_destination_label)
        
        repair_layout.addStretch()
        
        route_group_layout.addLayout(repair_layout)
        
        self.route_group.setLayout(route_group_layout)
        self.route_group.setVisible(False)  # Hidden until map and dataset ready
        controls_layout.addWidget(self.route_group)
        
        # ---- Progress Section (hidden by default) ----
        self.progress_group = QGroupBox("⏳ Download Progress")
        self.progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #333;
            }
        """)
        progress_group_layout = QVBoxLayout()
        progress_group_layout.setSpacing(8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
                border-radius: 4px;
            }
        """)
        progress_group_layout.addWidget(self.progress_bar)
        
        self.progress_status = QLabel("Ready")
        self.progress_status.setWordWrap(True)
        self.progress_status.setStyleSheet("color: #666; font-size: 11px;")
        progress_group_layout.addWidget(self.progress_status)
        
        self.progress_group.setLayout(progress_group_layout)
        self.progress_group.setVisible(False)  # Hidden by default
        controls_layout.addWidget(self.progress_group)
        
        # Add stretch before log to push it to the bottom
        controls_layout.addStretch()
        
        # ---- Log Section (at bottom, resizable) ----
        log_group = QGroupBox("📝 Activity Log")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #333;
            }
        """)
        log_group_layout = QVBoxLayout()
        log_group_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(80)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 10px;
                selection-background-color: #264f78;
            }
            QTextEdit QScrollBar:vertical {
                background-color: #2e2e2e;
                width: 12px;
                border-radius: 6px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 5px;
                min-height: 20px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QTextEdit QScrollBar::add-line:vertical,
            QTextEdit QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.log_text.setPlaceholderText("Activity log will appear here...")
        log_group_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_group_layout)
        controls_layout.addWidget(log_group)
        
        # Add stretch at bottom to push content up
        controls_layout.addStretch()
        
        # Set controls container as scroll area widget
        controls_scroll.setWidget(controls_container)
        content_layout.addWidget(controls_scroll, stretch=30)
        
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
        
        # Set maximum size to screen size
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.availableGeometry()
            self.setMaximumSize(screen_size.width(), screen_size.height())
    
    def browse_sumo_home(self):
        """Open file dialog to browse for SUMO_HOME directory."""
        current_path = self.sumo_home_input.text().strip() or DEFAULT_SUMO_HOME
        
        # Use non-native dialog for faster response
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select SUMO Installation Directory",
            current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks | QFileDialog.DontUseNativeDialog
        )
        
        if directory:
            self.sumo_home_input.setText(directory)
            self.verify_sumo_installation()
            self.log(f"SUMO_HOME set to: {directory}")
    
    def reset_sumo_home(self):
        """Reset SUMO_HOME to default path."""
        self.sumo_home_input.setText(DEFAULT_SUMO_HOME)
        self.verify_sumo_installation()
        self.log(f"SUMO_HOME reset to default: {DEFAULT_SUMO_HOME}")
    
    def verify_sumo_installation(self):
        """Verify the SUMO installation at the given path."""
        sumo_home = self.sumo_home_input.text().strip()
        
        if not sumo_home:
            self.sumo_status_label.setText("⚠️ Path is empty")
            self.sumo_status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            return False
        
        sumo_path = Path(sumo_home)
        
        if not sumo_path.exists():
            self.sumo_status_label.setText(f"❌ Path does not exist")
            self.sumo_status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            return False
        
        netconvert_path = sumo_path / 'bin' / 'netconvert'
        typemap_path = sumo_path / 'data' / 'typemap' / 'osmNetconvert.typ.xml'
        
        if netconvert_path.exists():
            if typemap_path.exists():
                self.sumo_status_label.setText(f"✅ SUMO found (netconvert + typemap)")
                self.sumo_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
            else:
                self.sumo_status_label.setText(f"✅ SUMO found (typemap missing)")
                self.sumo_status_label.setStyleSheet("color: #FF9800; font-size: 10px;")
            return True
        else:
            self.sumo_status_label.setText(f"⚠️ netconvert not found in {sumo_home}/bin/")
            self.sumo_status_label.setStyleSheet("color: #FF9800; font-size: 10px;")
            return False
    
    def on_sumo_home_changed(self, path: str):
        """Handle SUMO_HOME path change."""
        self.verify_sumo_installation()
        self.save_settings()
    
    def on_train_path_changed(self, path: str):
        """Handle train CSV path change."""
        self.validate_dataset_path(path, "train")
        self.save_settings()
        self.check_zones_visibility()
    
    def on_test_path_changed(self, path: str):
        """Handle test CSV path change."""
        self.validate_dataset_path(path, "test")
        self.save_settings()
        self.check_zones_visibility()
    
    def on_zones_changed(self, value: int):
        """Handle zones slider value change."""
        self.zones_value_label.setText(str(value))
        self.save_settings()
    
    def on_roads_junctions_filter_changed(self, state: int):
        """Handle roads and junctions filter checkbox change."""
        self.save_settings()
        
        # Reload the network with the new filter setting
        if self.network_parser:
            self.log(f"Filter changed: Roads and junctions only = {self.roads_junctions_only_checkbox.isChecked()}")
            # Re-render the network with current filter (async)
            self._reload_network_with_filter_async()
    
    def _reload_network_with_filter_async(self):
        """Reload the network view with current filter settings asynchronously."""
        if not self.network_parser:
            return
        
        # Show loading indicator and disable checkbox
        self.filter_loading_label.setText("⏳")
        self.roads_junctions_only_checkbox.setEnabled(False)
        
        # Force UI update before starting work
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Use QTimer to defer the actual work, allowing UI to update first
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._do_filter_reload)
    
    def _do_filter_reload(self):
        """Actually perform the filter reload (called after UI updates)."""
        if not self.network_parser:
            self._finish_filter_reload()
            return
        
        try:
            filter_roads_only = self.roads_junctions_only_checkbox.isChecked()
            
            # Load network with filter
            self.map_view.load_network(self.network_parser, roads_junctions_only=filter_roads_only)
            
            # Zoom to Porto city center
            self._zoom_to_porto_center()
            
            # Update status
            edges = self.network_parser.get_edges()
            
            if filter_roads_only:
                # Count filtered items
                filtered_edges = sum(1 for e in edges.values() if e.get('allows_passenger', True))
                junctions = self.network_parser.get_junctions()
                self.map_status_label.setText(
                    f"Map loaded (filtered): {filtered_edges} road edges, {len(junctions)} junctions"
                )
            else:
                nodes = self.network_parser.get_nodes()
                self.map_status_label.setText(
                    f"Map loaded: {len(edges)} edges, {len(nodes)} nodes"
                )
            self.map_status_label.setStyleSheet("color: #333; font-size: 12px;")
            
        except Exception as e:
            self.log(f"Error applying filter: {e}")
        finally:
            self._finish_filter_reload()
    
    def _finish_filter_reload(self):
        """Clean up after filter reload completes."""
        # Hide loading indicator and re-enable checkbox
        self.filter_loading_label.setText("")
        self.roads_junctions_only_checkbox.setEnabled(True)
    
    def on_satellite_changed(self, state: int):
        """Handle satellite imagery checkbox change."""
        self.save_settings()
        
        if self.network_parser:
            show_satellite = self.satellite_checkbox.isChecked()
            self.log(f"Satellite imagery: {'enabled' if show_satellite else 'disabled'}")
            
            # Show loading indicator
            if show_satellite:
                self.satellite_loading_label.setText("⏳")
                self.satellite_checkbox.setEnabled(False)
                
                # Connect to the finished signal (disconnect first to avoid duplicate connections)
                try:
                    self.map_view.satellite_loading_finished.disconnect(self._on_satellite_loading_finished)
                except RuntimeError:
                    pass  # Not connected yet
                self.map_view.satellite_loading_finished.connect(self._on_satellite_loading_finished)
                
                # Force UI update
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
                
                # Use QTimer to defer the work
                from PySide6.QtCore import QTimer
                QTimer.singleShot(50, self._do_satellite_toggle)
            else:
                # Disabling is quick, do it directly
                self.map_view.set_satellite_visible(False)
    
    def _do_satellite_toggle(self):
        """Actually toggle satellite visibility."""
        self.map_view.set_satellite_visible(True)
        # Note: loading indicator is cleared when satellite_loading_finished signal is emitted
    
    def _on_satellite_loading_finished(self):
        """Handle satellite loading completion."""
        self.satellite_loading_label.setText("")
        self.satellite_checkbox.setEnabled(True)
        self.log("Satellite imagery loaded")
    
    def is_map_ready(self) -> bool:
        """Check if the map is loaded."""
        return self.network_parser is not None
    
    def is_dataset_ready(self) -> bool:
        """Check if both train and test datasets are valid."""
        train_valid = self.train_valid_label.text() == "✅"
        test_valid = self.test_valid_label.text() == "✅"
        return train_valid and test_valid
    
    def check_zones_visibility(self):
        """Show zones and route sections only when map and dataset are ready."""
        if self.is_map_ready() and self.is_dataset_ready():
            if not self.zones_group.isVisible():
                self.zones_group.setVisible(True)
                self.route_group.setVisible(True)
                self.log("✅ Map and dataset ready - Zone and Route configuration enabled")
                # Load trip count from train dataset
                self.load_trip_count()
        else:
            self.zones_group.setVisible(False)
            self.route_group.setVisible(False)
    
    def check_resources(self):
        """Check if map and dataset resources exist."""
        self.log("Checking available resources...")
        
        # Check SUMO installation
        self.verify_sumo_installation()
        
        # Check for Porto network file
        project_path = Path(self.project_path)
        config_dir = project_path / 'config'
        net_file = config_dir / 'porto.net.xml'
        
        # Also check in Porto/config if project is in Porto folder
        porto_config = Path(self.project_path).parent.parent / 'Porto' / 'config'
        porto_net_file = porto_config / 'porto.net.xml'
        
        if net_file.exists():
            self.map_status.setText("✅ Network map available")
            self.map_status.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.download_map_btn.setVisible(False)  # Hide the button when map is available
            self.roads_junctions_only_checkbox.setVisible(True)  # Show the filter checkbox
            self.filter_loading_label.setVisible(False)
            self.satellite_checkbox.setVisible(True)  # Show the satellite checkbox
            self.satellite_loading_label.setVisible(False)
            self.load_network_async(net_file)
        elif porto_net_file.exists():
            self.map_status.setText("✅ Network map available (Porto folder)")
            self.map_status.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.download_map_btn.setVisible(False)  # Hide the button when map is available
            self.roads_junctions_only_checkbox.setVisible(True)  # Show the filter checkbox
            self.filter_loading_label.setVisible(False)
            self.satellite_checkbox.setVisible(True)  # Show the satellite checkbox
            self.satellite_loading_label.setVisible(False)
            self.load_network_async(porto_net_file)
        else:
            self.map_status.setText("❌ Network map not found\nClick to download from OpenStreetMap")
            self.map_status.setStyleSheet("color: #f44336; font-size: 11px;")
            self.download_map_btn.setVisible(True)
            self.download_map_btn.setEnabled(True)
            self.roads_junctions_only_checkbox.setVisible(False)
            self.filter_loading_label.setVisible(False)
            self.satellite_checkbox.setVisible(False)
            self.satellite_loading_label.setVisible(False)
            self.map_status_label.setText("Map not loaded - click 'Download & Render Map'")
        
        # Check dataset status
        self._check_dataset_status()
        
        self.log("Resource check complete")
    
    def _check_dataset_status(self):
        """Check dataset availability and set paths if found."""
        project_path = Path(self.project_path)
        dataset_dir = project_path / 'dataset'
        
        # Also check Porto/dataset
        porto_dataset_dir = Path(self.project_path).parent.parent / 'Porto' / 'dataset'
        
        # Check for train.csv
        train_file = dataset_dir / 'train.csv'
        porto_train = porto_dataset_dir / 'train.csv'
        
        if train_file.exists():
            self.train_path_input.setText(str(train_file))
            self.log(f"Train dataset found: {train_file}")
        elif porto_train.exists():
            self.train_path_input.setText(str(porto_train))
            self.log(f"Train dataset found: {porto_train}")
        else:
            self.train_status.setText("Select train.csv from Porto taxi dataset")
            self.train_status.setStyleSheet("color: #666; font-size: 9px;")
        
        # Check for test.csv
        test_file = dataset_dir / 'test.csv'
        porto_test = porto_dataset_dir / 'test.csv'
        
        if test_file.exists():
            self.test_path_input.setText(str(test_file))
            self.log(f"Test dataset found: {test_file}")
        elif porto_test.exists():
            self.test_path_input.setText(str(porto_test))
            self.log(f"Test dataset found: {porto_test}")
        else:
            self.test_status.setText("Select test.csv from Porto taxi dataset")
            self.test_status.setStyleSheet("color: #666; font-size: 9px;")
    
    def browse_dataset_file(self, file_type: str):
        """Open file dialog to browse for dataset CSV file.
        
        Args:
            file_type: Either 'train' or 'test'
        """
        # Get the appropriate input field
        if file_type == "train":
            input_field = self.train_path_input
            file_name = "train.csv"
        else:
            input_field = self.test_path_input
            file_name = "test.csv"
        
        # Determine start directory
        current_path = input_field.text().strip()
        
        if current_path:
            path_obj = Path(current_path)
            if path_obj.exists():
                start_dir = str(path_obj.parent)
            else:
                start_dir = self.project_path
        else:
            # Try to use the other file's directory if available
            other_input = self.test_path_input if file_type == "train" else self.train_path_input
            other_path = other_input.text().strip()
            if other_path and Path(other_path).exists():
                start_dir = str(Path(other_path).parent)
            else:
                start_dir = self.project_path
        
        # Use non-native dialog for faster response
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Porto Taxi Dataset ({file_name})",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        
        if file_path:
            input_field.setText(file_path)
            self.log(f"{file_type.capitalize()} dataset path set to: {file_path}")
    
    def validate_dataset_path(self, path: str, file_type: str):
        """Validate the dataset path and update status.
        
        Args:
            path: The file path to validate
            file_type: Either 'train' or 'test'
        """
        # Get the appropriate UI elements
        if file_type == "train":
            valid_label = self.train_valid_label
            status_label = self.train_status
            expected_name = "train.csv"
        else:
            valid_label = self.test_valid_label
            status_label = self.test_status
            expected_name = "test.csv"
        
        if not path:
            valid_label.setText("")
            status_label.setText(f"Select {expected_name} from Porto taxi dataset")
            status_label.setStyleSheet("color: #666; font-size: 9px;")
            return
        
        file_path = Path(path)
        
        if not file_path.exists():
            valid_label.setText("❌")
            status_label.setText("File does not exist")
            status_label.setStyleSheet("color: #f44336; font-size: 9px;")
            return
        
        if not file_path.is_file():
            valid_label.setText("❌")
            status_label.setText("Path is not a file")
            status_label.setStyleSheet("color: #f44336; font-size: 9px;")
            return
        
        if file_path.suffix.lower() != '.csv':
            valid_label.setText("⚠️")
            status_label.setText("File is not a CSV file")
            status_label.setStyleSheet("color: #FF9800; font-size: 9px;")
            return
        
        # Check file size and basic validation
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Quick check if it looks like the Porto dataset (check header)
            with open(file_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                
                # train.csv has POLYLINE column, test.csv may not
                if file_type == "train":
                    if 'POLYLINE' in header and 'TRIP_ID' in header:
                        valid_label.setText("✅")
                        status_label.setText(f"Valid ({size_mb:.1f} MB)")
                        status_label.setStyleSheet("color: #4CAF50; font-size: 9px;")
                        self.log(f"Train dataset validated: {file_path.name} ({size_mb:.1f} MB)")
                    else:
                        valid_label.setText("⚠️")
                        status_label.setText(f"CSV ({size_mb:.1f} MB) - missing expected columns")
                        status_label.setStyleSheet("color: #FF9800; font-size: 9px;")
                else:
                    # test.csv validation
                    if 'TRIP_ID' in header:
                        valid_label.setText("✅")
                        status_label.setText(f"Valid ({size_mb:.1f} MB)")
                        status_label.setStyleSheet("color: #4CAF50; font-size: 9px;")
                        self.log(f"Test dataset validated: {file_path.name} ({size_mb:.1f} MB)")
                    else:
                        valid_label.setText("⚠️")
                        status_label.setText(f"CSV ({size_mb:.1f} MB) - missing TRIP_ID")
                        status_label.setStyleSheet("color: #FF9800; font-size: 9px;")
        except Exception as e:
            valid_label.setText("❌")
            status_label.setText(f"Error: {str(e)[:25]}")
            status_label.setStyleSheet("color: #f44336; font-size: 9px;")
    
    def show_dataset_help(self):
        """Show help dialog with instructions for downloading the dataset."""
        help_text = """<h3>📊 Porto Taxi Dataset Download Instructions</h3>
        
<p>The Porto taxi trajectory dataset is available from Kaggle:</p>

<ol>
<li><b>Create a Kaggle account</b> (if you don't have one)<br/>
   Visit: <a href="https://www.kaggle.com">https://www.kaggle.com</a></li>
<br/>
<li><b>Go to the competition page:</b><br/>
   <a href="https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data">
   ECML/PKDD 15: Taxi Trajectory Prediction</a></li>
<br/>
<li><b>Download the dataset:</b><br/>
   Click on "Download All" or download <code>train.csv.zip</code></li>
<br/>
<li><b>Extract the file:</b><br/>
   Unzip <code>train.csv.zip</code> to get <code>train.csv</code></li>
<br/>
<li><b>Select the file:</b><br/>
   Use the browse button (📁) to select the extracted <code>train.csv</code> file</li>
</ol>

<p><b>Note:</b> The dataset is approximately 1.7 million taxi trips with GPS trajectories 
recorded in Porto, Portugal from July 2013 to June 2014.</p>

<p><b>File size:</b> ~1.6 GB (compressed) / ~8 GB (uncompressed)</p>"""
        
        QMessageBox.information(
            self,
            "Porto Taxi Dataset Help",
            help_text
        )
    
    def load_network_async(self, net_file: Path):
        """Load network asynchronously to prevent UI freeze."""
        if self.network_loader_worker and self.network_loader_worker.isRunning():
            return
        
        self.log(f"Loading network: {net_file.name}...")
        self.map_status_label.setText("🔄 Loading map...")
        self.map_status_label.setStyleSheet("color: #FF9800; font-size: 12px; font-weight: bold;")
        
        # Show loading overlay
        self.map_loading_progress.setValue(0)
        self.loading_label.setText("🗺️ Loading Map...")
        self.loading_status_label.setText("Initializing...")
        self.map_stack.setCurrentIndex(1)  # Show loading overlay
        
        self.network_loader_worker = NetworkLoaderWorker(str(net_file))
        self.network_loader_worker.progress.connect(self.on_network_load_progress)
        self.network_loader_worker.status.connect(self.on_network_load_status)
        self.network_loader_worker.finished.connect(self.on_network_load_finished)
        self.network_loader_worker.start()
    
    def on_network_load_progress(self, value: int):
        """Handle network loading progress."""
        self.map_loading_progress.setValue(value)
    
    def on_network_load_status(self, message: str):
        """Handle network loading status update."""
        self.loading_status_label.setText(message)
    
    def on_network_load_finished(self, success: bool, network_parser, message: str):
        """Handle network loading completion."""
        if success and network_parser:
            self.network_parser = network_parser
            
            # Apply filter setting when loading
            filter_roads_only = self.roads_junctions_only_checkbox.isChecked()
            self.map_view.load_network(self.network_parser, roads_junctions_only=filter_roads_only)
            
            # Zoom to Porto city center area
            self._zoom_to_porto_center()
            
            # Get network statistics
            edges = self.network_parser.get_edges()
            nodes = self.network_parser.get_nodes()
            bounds = self.network_parser.get_bounds()
            
            if filter_roads_only:
                # Count filtered items
                filtered_edges = sum(1 for e in edges.values() if e.get('allows_passenger', True))
                junctions = self.network_parser.get_junctions()
                self.map_status_label.setText(
                    f"Map loaded (filtered): {filtered_edges} road edges, {len(junctions)} junctions"
                )
                self.log(f"Network loaded (filtered): {filtered_edges} road edges, {len(junctions)} junctions")
            else:
                self.map_status_label.setText(
                    f"Map loaded: {len(edges)} edges, {len(nodes)} nodes"
                )
                self.log(f"Network loaded successfully: {len(edges)} edges, {len(nodes)} nodes")
            
            self.map_status_label.setStyleSheet("color: #333; font-size: 12px;")
            
            if bounds:
                self.log(f"Bounds: ({bounds['x_min']:.1f}, {bounds['y_min']:.1f}) to ({bounds['x_max']:.1f}, {bounds['y_max']:.1f})")
            
            # Show map view
            self.map_stack.setCurrentIndex(0)
            
            # Apply satellite setting if checked
            if self.satellite_checkbox.isChecked():
                self.map_view.set_satellite_visible(True)
            
            # Check if zones section should be shown
            self.check_zones_visibility()
        else:
            self.log(f"Error loading network: {message}")
            self.map_status_label.setText(f"Error loading map: {message}")
            self.map_status_label.setStyleSheet("color: #f44336; font-size: 12px;")
            self.loading_label.setText("❌ Loading Failed")
            self.loading_status_label.setText(message[:50] + "..." if len(message) > 50 else message)
    
    def _zoom_to_porto_center(self):
        """Zoom the map view to Porto city center."""
        if not self.network_parser:
            return
        
        bounds = self.network_parser.get_bounds()
        if not bounds:
            return
        
        # Calculate Porto city center area (roughly the downtown/central area)
        # The network bounds are in SUMO coordinates, we'll zoom to ~60% of the center
        x_range = bounds['x_max'] - bounds['x_min']
        y_range = bounds['y_max'] - bounds['y_min']
        
        # Center point
        center_x = (bounds['x_min'] + bounds['x_max']) / 2
        center_y = (bounds['y_min'] + bounds['y_max']) / 2
        
        # Create a rectangle around the center (about 40% of total area for better zoom)
        zoom_factor = 0.4
        half_width = (x_range * zoom_factor) / 2
        half_height = (y_range * zoom_factor) / 2
        
        zoom_rect = QRectF(
            center_x - half_width,
            center_y - half_height,
            half_width * 2,
            half_height * 2
        )
        
        # Fit the view to this rectangle
        self.map_view.fitInView(zoom_rect, Qt.KeepAspectRatio)
        self.map_view.current_zoom = 1.0  # Reset zoom tracking
    
    def _show_busy_indicator(self, message: str = "Processing..."):
        """Show busy indicator overlay on the map."""
        # Update busy label text
        self.busy_label.setText(f"⏳ {message}")
        # Show busy overlay (page 2)
        self.map_stack.setCurrentIndex(2)
        # Disable zoom buttons
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.zoom_default_btn.setEnabled(False)
        self.zoom_reset_btn.setEnabled(False)
        # Change cursor to wait
        from PySide6.QtGui import QCursor
        self.setCursor(QCursor(Qt.WaitCursor))
        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def _hide_busy_indicator(self):
        """Hide busy indicator and show map."""
        # Restore cursor
        self.unsetCursor()
        # Show map view (page 0)
        self.map_stack.setCurrentIndex(0)
        # Update status label
        if self.network_parser:
            edges = self.network_parser.get_edges()
            nodes = self.network_parser.get_nodes()
            self.map_status_label.setText(f"Map loaded: {len(edges)} edges, {len(nodes)} nodes")
            self.map_status_label.setStyleSheet("color: #333; font-size: 12px;")
        else:
            self.map_status_label.setText("No map loaded")
            self.map_status_label.setStyleSheet("color: #666; font-size: 12px;")
        # Re-enable zoom buttons
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.zoom_default_btn.setEnabled(True)
        self.zoom_reset_btn.setEnabled(True)
        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def zoom_in(self):
        """Zoom in on the map."""
        if self.network_parser:
            self._show_busy_indicator("Zooming in...")
            try:
                self.map_view.zoom_in()
                self.log("Zoomed in")
            finally:
                self._hide_busy_indicator()
    
    def zoom_out(self):
        """Zoom out on the map."""
        if self.network_parser:
            self._show_busy_indicator("Zooming out...")
            try:
                self.map_view.zoom_out()
                self.log("Zoomed out")
            finally:
                self._hide_busy_indicator()
    
    def zoom_to_default(self):
        """Zoom to the default view (Porto city center)."""
        if self.network_parser:
            self._show_busy_indicator("Zooming to city center...")
            try:
                self._zoom_to_porto_center()
                self.log("Zoomed to Porto city center")
            finally:
                self._hide_busy_indicator()
    
    def zoom_to_full(self):
        """Zoom to show the entire map."""
        if self.network_parser:
            self._show_busy_indicator("Zooming to full view...")
            try:
                self.map_view.zoom_fit(self.network_parser)
                self.log("Zoomed to full map view")
            finally:
                self._hide_busy_indicator()
    
    def download_map(self):
        """Start downloading the Porto map."""
        if self.download_worker and self.download_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A download is already in progress.")
            return
        
        # Verify SUMO installation first
        if not self.verify_sumo_installation():
            QMessageBox.warning(
                self, 
                "SUMO Not Found",
                "SUMO installation not found at the specified path.\n\n"
                "Please verify SUMO_HOME path is correct before downloading."
            )
            return
        
        # Determine output path
        project_path = Path(self.project_path)
        config_dir = project_path / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Get SUMO_HOME from input
        sumo_home = self.sumo_home_input.text().strip()
        
        self.log(f"Starting map download (SUMO_HOME: {sumo_home})...")
        self.download_map_btn.setEnabled(False)
        
        # Show progress section
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText("Starting download...")
        
        self.download_worker = DownloadWorker("map", str(config_dir), sumo_home=sumo_home)
        self.download_worker.progress.connect(self.on_progress)
        self.download_worker.status.connect(self.on_status)
        self.download_worker.finished.connect(self.on_map_download_finished)
        self.download_worker.start()
    
    def on_progress(self, value: int):
        """Handle progress update."""
        self.progress_bar.setValue(value)
    
    def on_status(self, message: str):
        """Handle status update."""
        self.progress_status.setText(message)
        self.log(message)
    
    def on_map_download_finished(self, success: bool, message: str):
        """Handle map download completion."""
        self.progress_status.setText("Complete" if success else "Failed")
        self.log(message)
        
        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Success", message)
            
            # Load the newly downloaded network
            project_path = Path(self.project_path)
            net_file = project_path / 'config' / 'porto.net.xml'
            if net_file.exists():
                self.map_status.setText("✅ Network map available")
                self.map_status.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
                self.download_map_btn.setVisible(False)  # Hide the button when map is available
                self.filter_checkbox_container.setVisible(True)  # Show the filter checkbox
                self.satellite_checkbox_container.setVisible(True)  # Show the satellite checkbox
                self.load_network_async(net_file)
            
            # Check dataset status
            self._check_dataset_status()
        else:
            self.progress_bar.setValue(0)
            QMessageBox.warning(self, "Download Failed", message)
            self.download_map_btn.setVisible(True)
            self.download_map_btn.setEnabled(True)
            self.roads_junctions_only_checkbox.setVisible(False)
            self.filter_loading_label.setVisible(False)
            self.satellite_checkbox.setVisible(False)
            self.satellite_loading_label.setVisible(False)
        
        # Hide progress section after a short delay
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, self.hide_progress_section)
    
    def hide_progress_section(self):
        """Hide the download progress section."""
        self.progress_group.setVisible(False)
    
    def log(self, message: str):
        """Add a message to the log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def get_settings_path(self) -> Path:
        """Get the path to the settings JSON file."""
        config_dir = Path(self.project_path) / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / SETTINGS_FILE
    
    def save_settings(self):
        """Save current settings to JSON file."""
        if self._loading_settings:
            return  # Don't save while loading
        
        settings = {
            'sumo_home': self.sumo_home_input.text().strip(),
            'train_csv_path': self.train_path_input.text().strip(),
            'test_csv_path': self.test_path_input.text().strip(),
            'num_zones': self.zones_slider.value(),
            'train_trip_count': getattr(self, '_train_trip_count', None),
            'roads_junctions_only': self.roads_junctions_only_checkbox.isChecked(),
            'show_satellite': self.satellite_checkbox.isChecked(),
            'fix_route': self.fix_route_checkbox.isChecked(),
            'fix_invalid_segments': self.fix_invalid_segments_checkbox.isChecked(),
        }
        
        try:
            settings_path = self.get_settings_path()
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Silently fail - settings are not critical
            print(f"Warning: Could not save settings: {e}")
    
    def load_settings(self):
        """Load settings from JSON file."""
        self._loading_settings = True
        
        try:
            settings_path = self.get_settings_path()
            
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Apply settings
                if 'sumo_home' in settings and settings['sumo_home']:
                    self.sumo_home_input.setText(settings['sumo_home'])
                
                if 'train_csv_path' in settings and settings['train_csv_path']:
                    self.train_path_input.setText(settings['train_csv_path'])
                
                if 'test_csv_path' in settings and settings['test_csv_path']:
                    self.test_path_input.setText(settings['test_csv_path'])
                
                if 'num_zones' in settings:
                    self.zones_slider.setValue(settings['num_zones'])
                    self.zones_value_label.setText(str(settings['num_zones']))
                
                if 'train_trip_count' in settings and settings['train_trip_count']:
                    self._train_trip_count = settings['train_trip_count']
                
                if 'roads_junctions_only' in settings:
                    self.roads_junctions_only_checkbox.setChecked(settings['roads_junctions_only'])
                
                if 'show_satellite' in settings:
                    self.satellite_checkbox.setChecked(settings['show_satellite'])
                
                if 'fix_route' in settings:
                    self.fix_route_checkbox.setChecked(settings['fix_route'])
                
                if 'fix_invalid_segments' in settings:
                    self.fix_invalid_segments_checkbox.setChecked(settings['fix_invalid_segments'])
                
                self.log("Settings loaded from config")
        except Exception as e:
            # Silently fail - use defaults
            print(f"Warning: Could not load settings: {e}")
        finally:
            self._loading_settings = False
    
    def load_trip_count(self):
        """Load or count the number of trips in the train dataset."""
        # Check if we have cached count
        if hasattr(self, '_train_trip_count') and self._train_trip_count:
            self._update_trip_count_ui(self._train_trip_count)
            return
        
        train_path = self.train_path_input.text().strip()
        if not train_path or not Path(train_path).exists():
            self.trips_count_label.setText("N/A")
            return
        
        self.trips_count_label.setText("Counting...")
        self.log("Counting trips in train dataset...")
        
        # Count trips in background
        from PySide6.QtCore import QThread, Signal
        
        class TripCountWorker(QThread):
            finished = Signal(int)
            
            def __init__(self, file_path):
                super().__init__()
                self.file_path = file_path
            
            def run(self):
                try:
                    count = 0
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        # Skip header
                        next(f, None)
                        for _ in f:
                            count += 1
                    self.finished.emit(count)
                except Exception as e:
                    print(f"Error counting trips: {e}")
                    self.finished.emit(0)
        
        self._trip_count_worker = TripCountWorker(train_path)
        self._trip_count_worker.finished.connect(self._on_trip_count_finished)
        self._trip_count_worker.start()
    
    def _on_trip_count_finished(self, count: int):
        """Handle trip count completion."""
        self._train_trip_count = count
        self._update_trip_count_ui(count)
        self.save_settings()  # Cache the count
        self.log(f"Train dataset contains {count:,} trips")
    
    def _update_trip_count_ui(self, count: int):
        """Update UI with trip count."""
        self.trips_count_label.setText(f"{count:,}")
        self.route_spinbox.setMaximum(count if count > 0 else 1)
        self.route_info_label.setText(f"Select a route from 1 to {count:,}")
    
    def _on_fix_route_changed(self):
        """Handle fix route checkbox change."""
        self.save_settings()
        # Refresh the route display if a route is currently shown
        if hasattr(self, '_route_items') and self._route_items:
            self.show_selected_route()
    
    def show_selected_route(self):
        """Display the selected route on the map."""
        route_num = self.route_spinbox.value()
        train_path = self.train_path_input.text().strip()
        
        if not train_path or not Path(train_path).exists():
            QMessageBox.warning(self, "Error", "Train dataset not found.")
            return
        
        if not self.network_parser:
            QMessageBox.warning(self, "Error", "Map not loaded.")
            return
        
        self.log(f"Loading route #{route_num}...")
        self.route_info_label.setText(f"Loading route #{route_num}...")
        
        # Load the specific trip's polyline
        try:
            polyline = self._load_trip_polyline(train_path, route_num)
            
            if polyline and len(polyline) >= 2:
                # Detect real start and end points
                real_start_idx, real_end_idx = self._detect_real_start_and_end(polyline)
                
                # Update labels (without coordinates)
                if real_start_idx < len(polyline):
                    self.real_start_label.setText(f"Real Start: Point {real_start_idx + 1}")
                else:
                    self.real_start_label.setText("Real Start: N/A")
                
                if real_end_idx < len(polyline):
                    self.real_destination_label.setText(f"Real Destination: Point {real_end_idx + 1}")
                else:
                    self.real_destination_label.setText("Real Destination: N/A")
                
                # Store original length
                original_length = len(polyline)
                
                # Apply route repair if checkbox is checked
                if self.fix_route_checkbox.isChecked():
                    polyline = polyline[real_start_idx:real_end_idx + 1]
                
                # Split at invalid segments if checkbox is checked
                segments = [polyline]  # Default: single segment
                if self.fix_invalid_segments_checkbox.isChecked():
                    segments = self._split_at_invalid_segments(polyline)
                    # Apply trim start/end to each segment
                    trimmed_segments = []
                    for segment in segments:
                        if len(segment) >= 2:
                            seg_start, seg_end = self._detect_real_start_and_end(segment)
                            trimmed_seg = segment[seg_start:seg_end + 1]
                            if len(trimmed_seg) >= 2:  # Only add if has at least 2 points
                                trimmed_segments.append(trimmed_seg)
                    segments = trimmed_segments if trimmed_segments else segments
                
                # Clear previous route
                self.clear_route_display()
                
                # Get validation result before splitting (for invalid segments info)
                original_validation = validate_trip_segments(polyline) if polyline else None
                
                # Draw route on map (returns validation result)
                validation_result = self._draw_route_on_map(
                    segments, 
                    route_num, 
                    show_invalid_in_red=not self.fix_invalid_segments_checkbox.isChecked(),
                    original_polyline=polyline if not self.fix_invalid_segments_checkbox.isChecked() else None
                )
                
                # Build info text with validation status
                repair_info = ""
                if self.fix_route_checkbox.isChecked() and original_length != len(polyline):
                    repair_info = f" (Trimmed: {original_length} → {len(polyline)} points)"
                
                # Build route info text
                if self.fix_invalid_segments_checkbox.isChecked() and len(segments) > 1:
                    total_points = sum(len(seg) for seg in segments)
                    route_info_lines = [
                        f"Route #{route_num}: {len(polyline)} points{repair_info}",
                        f"Invalid segments: {original_validation.invalid_segment_count if original_validation else 0}",
                        f"New routes: {len(segments)} segments ({total_points} total points)"
                    ]
                    self.route_info_label.setText("\n".join(route_info_lines))
                    self.route_info_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-weight: bold;")
                elif validation_result and not validation_result.is_valid:
                    self.route_info_label.setText(
                        f"Route #{route_num}: {len(polyline)} points{repair_info} | "
                        f"⚠️ {validation_result.invalid_segment_count} invalid segment(s)"
                    )
                    self.route_info_label.setStyleSheet("color: #f44336; font-size: 9px; font-weight: bold;")
                else:
                    self.route_info_label.setText(
                        f"Route #{route_num}: {len(polyline)} GPS points{repair_info} ✅"
                    )
                    self.route_info_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-weight: bold;")
                
                log_msg = f"Route #{route_num} displayed"
                if self.fix_invalid_segments_checkbox.isChecked():
                    log_msg += f" as {len(segments)} segment(s)"
                else:
                    log_msg += f" with {len(polyline)} GPS points"
                if repair_info:
                    log_msg += f" (trimmed from {original_length})"
                self.log(log_msg)
            else:
                self.route_info_label.setText(f"Route #{route_num}: No valid GPS data")
                self.route_info_label.setStyleSheet("color: #666; font-size: 9px;")
                self.log(f"Route #{route_num} has no valid GPS data")
        except Exception as e:
            self.route_info_label.setText(f"Error loading route: {str(e)[:30]}")
            self.route_info_label.setStyleSheet("color: #f44336; font-size: 9px;")
            self.log(f"Error loading route #{route_num}: {e}")
    
    def _detect_real_start_and_end(self, polyline: list) -> Tuple[int, int]:
        """
        Detect real start and end points by finding where static points end.
        
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
    
    def _split_at_invalid_segments(self, polyline: list) -> list:
        """
        Split polyline at invalid segments (>1000m).
        
        Args:
            polyline: List of GPS coordinates
            
        Returns:
            List of polyline segments (each segment is a list of points)
        """
        if not polyline or len(polyline) < 2:
            return [polyline] if polyline else []
        
        # Validate to find invalid segments
        validation_result = validate_trip_segments(polyline)
        invalid_indices = set(validation_result.invalid_segment_indices)
        
        if not invalid_indices:
            # No invalid segments, return original as single segment
            return [polyline]
        
        # Split at invalid segments
        segments = []
        current_segment = [polyline[0]]  # Start with first point
        
        for i in range(1, len(polyline)):
            # If previous segment (i-1 to i) is invalid, start a new segment
            if (i - 1) in invalid_indices:
                # End current segment (don't include the point after invalid segment)
                if len(current_segment) > 0:
                    segments.append(current_segment)
                # Start new segment with current point
                current_segment = [polyline[i]]
            else:
                # Continue current segment
                current_segment.append(polyline[i])
        
        # Add the last segment
        if len(current_segment) > 0:
            segments.append(current_segment)
        
        return segments
    
    def _load_trip_polyline(self, csv_path: str, trip_num: int) -> list:
        """Load a specific trip's polyline from the CSV file."""
        import ast
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f, None)
            
            for i, line in enumerate(f, 1):
                if i == trip_num:
                    # Parse the line to extract POLYLINE column (last column)
                    # CSV format: "TRIP_ID","CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","TAXI_ID","TIMESTAMP","DAY_TYPE","MISSING_DATA","POLYLINE"
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
                        print(f"Error parsing polyline: {e}")
                        return []
        return []
    
    def _draw_route_on_map(self, segments_or_polyline, route_num: int, show_invalid_in_red: bool = False, original_polyline: list = None):
        """Draw the route on the map with colored points and validated segments.
        
        Args:
            segments_or_polyline: Either a single polyline (list) or list of segments (list of lists)
            route_num: Route number for display
            show_invalid_in_red: If True, show invalid segments in red (when not fixing)
            original_polyline: Original polyline before splitting (for invalid segment detection)
        """
        from PySide6.QtGui import QBrush, QColor, QFont, QPen
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem

        # Store route items for later clearing
        self._route_items = []
        
        # Handle both single polyline and multiple segments
        # Check if first element is a list of lists (multiple segments) or a single coordinate pair
        if (segments_or_polyline and 
            len(segments_or_polyline) > 0 and 
            isinstance(segments_or_polyline[0], list) and 
            len(segments_or_polyline[0]) > 0 and
            isinstance(segments_or_polyline[0][0], (list, tuple)) and
            len(segments_or_polyline[0][0]) == 2):
            # Multiple segments (list of polylines)
            segments = segments_or_polyline
            is_multiple_segments = True
        else:
            # Single polyline
            segments = [segments_or_polyline]
            is_multiple_segments = False
        
        # Get invalid segment indices if showing invalid in red
        invalid_segment_set = set()
        if show_invalid_in_red and original_polyline:
            validation_result = validate_trip_segments(original_polyline)
            invalid_segment_set = set(validation_result.invalid_segment_indices)
        
        # Get Y bounds for flipping (to match the flipped network map)
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        # Different colors for each segment
        segment_colors = [
            QColor(33, 150, 243),   # Blue
            QColor(76, 175, 80),    # Green
            QColor(255, 152, 0),    # Orange
            QColor(156, 39, 176),   # Purple
            QColor(244, 67, 54),    # Red
            QColor(0, 188, 212),    # Cyan
            QColor(255, 235, 59),   # Yellow
            QColor(121, 85, 72),    # Brown
        ]
        
        all_sumo_points = []  # For zoom calculation
        point_counter = 1  # Global point counter across all segments
        
        # Process each segment
        for seg_idx, polyline in enumerate(segments):
            if not polyline or len(polyline) < 2:
                continue
            
            # Validate the trip segments
            validation_result = validate_trip_segments(polyline)
            
            # Log validation results
            if not validation_result.is_valid:
                self.log(f"⚠️ Route #{route_num} Segment {seg_idx + 1} has {validation_result.invalid_segment_count} invalid segment(s) (>1000m)")
            
            # Convert GPS coordinates to SUMO coordinates
            sumo_points = []
            for lon, lat in polyline:
                result = self.network_parser.gps_to_sumo_coords(lon, lat)
                if result is not None:
                    sumo_x, sumo_y = result
                    # Flip Y to match the flipped network map
                    sumo_points.append((sumo_x, flip_y(sumo_y)))
            
            if not sumo_points:
                continue
            
            all_sumo_points.extend(sumo_points)
            
            # Get color for this segment
            segment_color = segment_colors[seg_idx % len(segment_colors)]
            line_color = QColor(segment_color.red(), segment_color.green(), segment_color.blue(), 200)
            
            # Draw lines connecting points
            # For single polyline with invalid segments, need to map indices correctly
            segment_start_idx = sum(len(segments[j]) for j in range(seg_idx)) if is_multiple_segments else 0
            
            for i in range(len(sumo_points) - 1):
                x1, y1 = sumo_points[i]
                x2, y2 = sumo_points[i + 1]
                
                # Determine if this segment is invalid (only for single polyline mode)
                is_invalid = False
                if show_invalid_in_red and not is_multiple_segments:
                    # Check if segment i is invalid in original polyline
                    original_idx = segment_start_idx + i
                    is_invalid = original_idx in invalid_segment_set
                
                # Use red for invalid segments, segment color for valid
                if is_invalid:
                    line_color_use = QColor(244, 67, 54, 220)  # Red for invalid
                    line_width = 8
                else:
                    line_color_use = line_color
                    line_width = 6
                
                line = self.map_view.scene.addLine(
                    x1, y1, x2, y2,
                    QPen(line_color_use, line_width)
                )
                line.setZValue(5)  # Above network, below points
                self._route_items.append(line)
            
            # Point size and font settings
            point_size = 60
            font = QFont("Arial", 20, QFont.Bold)
            
            # Draw points with numbers
            for i, (x, y) in enumerate(sumo_points):
                # Determine color based on position within segment
                if i == 0:
                    # First point of segment - Green
                    color = QColor(76, 175, 80)  # Green
                elif i == len(sumo_points) - 1:
                    # Last point of segment - Red
                    color = QColor(244, 67, 54)  # Red
                else:
                    # Middle points - Use segment color
                    color = segment_color
                
                # Create point circle
                ellipse = QGraphicsEllipseItem(
                    x - point_size/2, y - point_size/2,
                    point_size, point_size
                )
                ellipse.setBrush(QBrush(color))
                ellipse.setPen(QPen(QColor(255, 255, 255), 3))  # White border
                ellipse.setZValue(10)  # On top
                self.map_view.scene.addItem(ellipse)
                self._route_items.append(ellipse)
                
                # Create point number text
                text = QGraphicsTextItem(str(point_counter))
                text.setFont(font)
                text.setDefaultTextColor(QColor(255, 255, 255))
                # Center text on point
                text_rect = text.boundingRect()
                text.setPos(x - text_rect.width()/2, y - text_rect.height()/2)
                text.setZValue(11)  # Above points
                self.map_view.scene.addItem(text)
                self._route_items.append(text)
                
                point_counter += 1
        
        # Zoom to fit all segments
        if all_sumo_points:
            xs = [p[0] for p in all_sumo_points]
            ys = [p[1] for p in all_sumo_points]
            margin = 100  # Add margin around route
            from PySide6.QtCore import QRectF
            rect = QRectF(
                min(xs) - margin, min(ys) - margin,
                max(xs) - min(xs) + 2*margin, max(ys) - min(ys) + 2*margin
            )
            self.map_view.fitInView(rect, Qt.KeepAspectRatio)
        
        # Return validation result for UI updates (from first segment or combined)
        if segments:
            return validate_trip_segments(segments[0])
        return None
    
    def clear_route_display(self):
        """Clear the displayed route from the map."""
        if hasattr(self, '_route_items') and self._route_items:
            for item in self._route_items:
                self.map_view.scene.removeItem(item)
            self._route_items = []
            self.route_info_label.setText("Route cleared")
            self.log("Route display cleared")

