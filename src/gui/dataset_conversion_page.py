"""
Dataset Conversion page for Porto taxi data.
Provides map visualization and dataset conversion controls.
"""

import json
import math
import os
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QGraphicsPolygonItem,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QStackedWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.route_finding import (EdgeSpatialIndex, build_edges_data,
                                     build_node_positions,
                                     compute_green_orange_edges,
                                     project_point_onto_polyline,
                                     project_point_onto_polyline_with_segment,
                                     shortest_path_dijkstra)
from src.utils.dataset_conversion_mp import run_multiprocess
from src.utils.trajectory_converter import (
    apply_gps_offset,
    convert_trajectory,
    iter_trajectories_from_csv,
)
from src.utils.trip_validator import (TripValidationResult,
                                      validate_trip_segments)

# Default SUMO_HOME path
DEFAULT_SUMO_HOME = "/usr/share/sumo"

# Settings file name
SETTINGS_FILE = "porto_settings.json"

# Porto bounding box coordinates (updated to match the new centered network file)
# Extracted from porto.net.xml origBoundary: lon_min,lat_min,lon_max,lat_max
PORTO_BBOX = {
    'north': 41.271161,   # max_lat
    'south': 41.071545,   # min_lat
    'east': -8.295034,    # max_lon
    'west': -8.716066     # min_lon
}


class EdgeSpatialIndex:
    """
    Grid-based spatial index for fast edge lookups.
    Divides the network into a grid and indexes edges by their bounding boxes.
    Lane # is always ignored - edges are indexed by base edge ID only.
    """
    
    def __init__(self, edges_dict: dict, cell_size: float = 500.0):
        """
        Initialize spatial index.
        
        Args:
            edges_dict: Dictionary of edge_id -> edge_data
            cell_size: Size of each grid cell in meters (default 500m)
        """
        self.cell_size = cell_size
        self.grid = defaultdict(set)  # (grid_x, grid_y) -> set of edge_ids
        self.edge_bounds = {}  # edge_id -> (min_x, min_y, max_x, max_y)
        
        # Build index
        # Important: Deduplicate by base edge ID (ignore lane #)
        base_edge_map = {}  # base_id -> (edge_id, edge_data) - keep first variant
        
        for edge_id, edge_data in edges_dict.items():
            # Get base edge ID (strip # suffix)
            base_id = edge_id.split('#')[0] if '#' in edge_id else edge_id
            
            # Keep first variant for each base edge
            if base_id not in base_edge_map:
                base_edge_map[base_id] = (edge_id, edge_data)
        
        # Index only base edges (one variant per base edge)
        for base_id, (edge_id, edge_data) in base_edge_map.items():
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Calculate bounding box
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            self.edge_bounds[edge_id] = (min_x, min_y, max_x, max_y)
            
            # Add edge to all grid cells it intersects
            min_grid_x = int(min_x / cell_size)
            max_grid_x = int(max_x / cell_size)
            min_grid_y = int(min_y / cell_size)
            max_grid_y = int(max_y / cell_size)
            
            for grid_x in range(min_grid_x, max_grid_x + 1):
                for grid_y in range(min_grid_y, max_grid_y + 1):
                    self.grid[(grid_x, grid_y)].add(edge_id)
    
    def get_candidates_in_radius(self, x: float, y: float, radius: float) -> set:
        """
        Get all edge IDs that might be within radius (using bounding box check).
        
        Args:
            x: X coordinate in SUMO space
            y: Y coordinate in SUMO space
            radius: Search radius in meters
        
        Returns:
            Set of edge IDs that might be within radius
        """
        # Calculate grid cells to check
        min_x = x - radius
        max_x = x + radius
        min_y = y - radius
        max_y = y + radius
        
        min_grid_x = int(min_x / self.cell_size)
        max_grid_x = int(max_x / self.cell_size)
        min_grid_y = int(min_y / self.cell_size)
        max_grid_y = int(max_y / self.cell_size)
        
        candidates = set()
        for grid_x in range(min_grid_x, max_grid_x + 1):
            for grid_y in range(min_grid_y, max_grid_y + 1):
                candidates.update(self.grid.get((grid_x, grid_y), set()))
        
        return candidates


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


def _load_trip_polyline_static(csv_path: str, trip_num: int) -> list:
    """Load a specific trip's polyline from the CSV file. Thread-safe."""
    import ast
    with open(csv_path, 'r', encoding='utf-8') as f:
        next(f, None)  # Skip header
        for i, line in enumerate(f, 1):
            if i == trip_num:
                try:
                    polyline_start = line.rfind('"[[')
                    if polyline_start == -1:
                        polyline_start = line.rfind('"[]')
                    if polyline_start != -1:
                        polyline_str = line[polyline_start+1:].strip().rstrip('"')
                        return ast.literal_eval(polyline_str)
                except Exception:
                    pass
                return []
    return []


def _load_trip_row_static(csv_path: str, trip_num: int) -> Tuple[Optional[List[List[float]]], Optional[int]]:
    """Load polyline and TIMESTAMP for a trip from CSV. Thread-safe. Returns (polyline, timestamp) or (None, None)."""
    import ast
    import csv as csv_module
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_module.reader(f)
            header = next(reader, None)
            if not header:
                return None, None
            ts_idx = None
            for i, h in enumerate(header):
                if h.strip('"') == 'TIMESTAMP':
                    ts_idx = i
                    break
            for i, row in enumerate(reader, 1):
                if i == trip_num:
                    timestamp = None
                    if ts_idx is not None and ts_idx < len(row):
                        try:
                            timestamp = int(str(row[ts_idx]).strip('"'))
                        except (ValueError, TypeError):
                            pass
                    polyline_str = None
                    for cell in row:
                        s = str(cell).strip()
                        if s.startswith('[[') and s.endswith(']]'):
                            polyline_str = s
                            break
                    if polyline_str:
                        polyline = ast.literal_eval(polyline_str)
                        if isinstance(polyline, list) and len(polyline) >= 2:
                            return polyline, timestamp
                    return None, None
    except Exception:
        pass
    return None, None


def _prepare_route_data_static(
    original_polyline: List[List[float]],
) -> Tuple[object, int, int, list]:
    """Prepare route data (validation, trim indices). Thread-safe."""
    validation_result = validate_trip_segments(original_polyline)
    real_start_idx, real_end_idx = _detect_real_start_and_end_static(original_polyline)
    segment_trim_data = []
    if validation_result.invalid_segment_count > 0:
        split_segments = _split_at_invalid_segments_static(original_polyline)
        for segment in split_segments:
            if len(segment) >= 2:
                seg_start, seg_end = _detect_real_start_and_end_static(segment)
                segment_trim_data.append((seg_start, seg_end))
    return validation_result, real_start_idx, real_end_idx, segment_trim_data


def _detect_real_start_and_end_static(polyline: list) -> Tuple[int, int]:
    """Detect real start and end indices. Thread-safe."""
    if not polyline or len(polyline) < 3:
        return 0, len(polyline) - 1 if polyline else 0
    STATIC_THRESHOLD = 15.0
    R = 6371000

    def haversine_m(coord1, coord2):
        lat1, lon1 = coord1[0], coord1[1]
        lat2, lon2 = coord2[0], coord2[1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    real_start = 0
    for i in range(len(polyline) - 1):
        if haversine_m(polyline[i], polyline[i + 1]) > STATIC_THRESHOLD:
            real_start = i
            break
    real_end = len(polyline) - 1
    for i in range(len(polyline) - 1, 0, -1):
        if haversine_m(polyline[i - 1], polyline[i]) > STATIC_THRESHOLD:
            real_end = i
            break
    return real_start, real_end


def _split_at_invalid_segments_static(polyline: list) -> list:
    """Split polyline at invalid segments. Thread-safe."""
    if not polyline or len(polyline) < 2:
        return [polyline] if polyline else []
    validation_result = validate_trip_segments(polyline)
    invalid_indices = set(validation_result.invalid_segment_indices)
    if not invalid_indices:
        return [polyline]
    segments = []
    current_segment = [polyline[0]]
    for i in range(1, len(polyline)):
        if i - 1 in invalid_indices:
            segments.append(current_segment)
            current_segment = [polyline[i]]
        else:
            current_segment.append(polyline[i])
    if current_segment:
        segments.append(current_segment)
    return segments


class RouteLoadWorker(QThread):
    """Worker thread for loading route polyline and preparing route data."""

    finished = Signal(bool, object, str)  # success, result_tuple or None, error_msg

    def __init__(self, train_path: str, route_num: int, parent=None):
        super().__init__(parent)
        self.train_path = train_path
        self.route_num = route_num

    def run(self):
        try:
            original_polyline = _load_trip_polyline_static(self.train_path, self.route_num)
            if not original_polyline or len(original_polyline) < 2:
                self.finished.emit(False, None, "No valid GPS data")
                return
            invalid_segments, real_start_idx, real_end_idx, segment_trim_data = _prepare_route_data_static(
                original_polyline
            )
            result = (original_polyline, invalid_segments, real_start_idx, real_end_idx, segment_trim_data)
            self.finished.emit(True, result, "")
        except Exception as e:
            self.finished.emit(False, None, str(e))


class DatasetGenerationWorker(QThread):
    """Worker thread for batch dataset generation - exports trajectories with SUMO routes to JSON.
    Uses shared trajectory_converter logic. Supports multiprocessing for speed.
    """

    progress = Signal(int, int, str)  # current, total, status
    finished = Signal(int, int, str)  # saved_count, total_processed, error_msg (empty if ok)

    def __init__(
        self,
        train_path: str,
        output_path: str,
        start_traj: int,
        last_traj: int,
        network_path: str,
        network_parser,
        y_min: float,
        y_max: float,
        workers: int,
        offset_x: float,
        offset_y: float,
        cancelled_callback,
        parent=None,
    ):
        super().__init__(parent)
        self.train_path = train_path
        self.output_path = output_path
        self.start_traj = start_traj
        self.last_traj = last_traj
        self.network_path = network_path
        self.network_parser = network_parser
        self.y_min = y_min
        self.y_max = y_max
        self.workers = max(1, workers)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._cancelled = cancelled_callback

    def run(self):
        total = max(0, self.last_traj - self.start_traj + 1)
        if total == 0:
            self.finished.emit(0, 0, "")
            return

        if self.workers > 1:
            self._run_multiprocess(total)
        else:
            self._run_single(total)

    def _run_multiprocess(self, total: int):
        """Use multiprocessing for parallel conversion."""
        trajectories = list(
            iter_trajectories_from_csv(self.train_path, self.start_traj, self.last_traj)
        )
        if not trajectories:
            self.finished.emit(0, 0, "")
            return

        last_emitted = [0]  # use list to allow closure to mutate

        def on_progress(current: int, tot: int):
            if self._cancelled():
                return
            # Throttle: emit every 5 trajectories or when done
            if current == 1 or current - last_emitted[0] >= 5 or current == tot:
                last_emitted[0] = current
                pct = 100 * current // tot if tot else 0
                self.progress.emit(current, tot, f"Processing... {current}/{tot} ({pct}%)")

        saved_count, total_processed = run_multiprocess(
            trajectories,
            self.network_path,
            self.output_path,
            self.workers,
            use_polygon=False,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            progress_callback=on_progress,
            cancelled_callback=self._cancelled,
        )
        err = "Cancelled" if self._cancelled() else ""
        self.finished.emit(saved_count, total_processed, err)

    def _run_single(self, total: int):
        """Single-threaded conversion (supports per-trajectory cancellation)."""
        saved_count = 0
        total_processed = 0
        edges_data = build_edges_data(self.network_parser)
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        node_positions = build_node_positions(self.network_parser)
        spatial_index = EdgeSpatialIndex(edges_data, cell_size=500.0)
        os.makedirs(self.output_path, exist_ok=True)

        iterator = iter_trajectories_from_csv(self.train_path, self.start_traj, self.last_traj)
        last_emitted = [0]
        for current, (trip_num, polyline, base_timestamp) in enumerate(iterator, 1):
            if self._cancelled():
                self.finished.emit(saved_count, total_processed, "Cancelled")
                return

            # Throttle progress: emit every 5 trajectories or when done
            if current == 1 or current - last_emitted[0] >= 5 or current == total:
                last_emitted[0] = current
                self.progress.emit(current, total, f"Processing trajectory {trip_num}...")

            rec = convert_trajectory(
                trip_num,
                polyline,
                base_timestamp,
                self.network_parser,
                edges_data,
                edge_shapes,
                node_positions,
                self.y_min,
                self.y_max,
                use_polygon=False,
                offset_x=self.offset_x,
                offset_y=self.offset_y,
                spatial_index=spatial_index,
                cancelled_callback=self._cancelled,
            )
            if rec:
                out_file = Path(self.output_path) / f"traj_{trip_num}.json"
                try:
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=2)
                    saved_count += 1
                except Exception:
                    pass
            total_processed += 1

        self.finished.emit(saved_count, total_processed, "")


class DatasetConversionPage(QWidget):
    """Page for Porto dataset conversion with map view."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.network_parser = None
        self.network_file_path = None  # Store network file path for TraCI routing
        self.sumo_net = None  # Cached sumolib network object (loaded once)
        self._edge_spatial_index = None  # Spatial index for fast edge lookups
        self.destination_candidate_edges = []  # Store destination candidate edges from Step 1.4
        
        # Candidate edge counts
        self.START_END_CANDIDATES = 10  # Number of candidate edges for start and end points
        self.INTERMEDIATE_CANDIDATES = 6  # Number of candidate edges for intermediate GPS points
        self.download_worker = None
        self.network_loader_worker = None
        self._route_load_worker = None
        self._loading_settings = False  # Flag to prevent save during load
        self._save_settings_timer = QTimer(self)
        self._save_settings_timer.setSingleShot(True)
        self._save_settings_timer.timeout.connect(self.save_settings)
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.timeout.connect(self._on_validation_timer_fired)
        self._validation_args = None  # (path, file_type) for debounced validation
        self._train_trip_count = None  # Cached trip count
        self._route_items = []  # Graphics items for current route display
        self._candidate_edge_items = []  # Green/orange edge items (separate for fast toggle)
        self._dataset_gen_worker = None
        self._dataset_gen_cancelled = False
        self._sumo_route_items = []  # SUMO path lines and stars (separate for fast toggle)
        self._current_polyline = None  # Store current polyline for SUMO route mapping
        self._current_segments = None  # Store current segments for SUMO route mapping
        self._cached_edges_data = None  # Cached for fast candidate edge drawing
        
        # Cache for prepared route data (computed only once per route selection)
        self._cached_route_num = None
        self._cached_original_polyline = None
        self._cached_route_data = None  # Tuple: (invalid_segments, real_start_idx, real_end_idx, segment_trim_data)
        
        self.init_ui()
        
        # Load saved settings
        self.load_settings()
        
        # Use QTimer to check resources after UI is shown (non-blocking)
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
        
        # OSM map checkbox
        self.osm_map_checkbox = QCheckBox("Show OSM map")
        self.osm_map_checkbox.setToolTip(
            "When checked, display OpenStreetMap tiles as map background\n"
            "(downloads tiles from OpenStreetMap service)"
        )
        self.osm_map_checkbox.setStyleSheet("""
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
        self.osm_map_checkbox.stateChanged.connect(self.on_osm_map_changed)
        self.osm_map_checkbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.osm_map_checkbox)
        
        # Loading indicator for OSM map
        self.osm_map_loading_label = QLabel("")
        self.osm_map_loading_label.setStyleSheet("font-size: 14px;")
        self.osm_map_loading_label.setFixedWidth(20)
        self.osm_map_loading_label.setVisible(False)
        map_header_layout.addWidget(self.osm_map_loading_label)
        
        map_header_layout.addSpacing(10)
        
        # SUMO network checkbox
        self.show_network_checkbox = QCheckBox("Show SUMO network")
        self.show_network_checkbox.setToolTip(
            "When checked, display the SUMO network (edges and nodes)\n"
            "on top of the map"
        )
        self.show_network_checkbox.setStyleSheet("""
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
        self.show_network_checkbox.setChecked(True)  # Show by default
        self.show_network_checkbox.stateChanged.connect(self.on_show_network_changed)
        self.show_network_checkbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.show_network_checkbox)
        
        map_header_layout.addSpacing(10)
        
        # Map offset controls
        self.map_offset_label = QLabel("GPS offset:")
        self.map_offset_label.setStyleSheet("color: #333; font-size: 11px;")
        self.map_offset_label.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.map_offset_label)
        
        # X offset
        self.map_offset_x_label = QLabel("X:")
        self.map_offset_x_label.setStyleSheet("color: #333; font-size: 10px;")
        self.map_offset_x_label.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.map_offset_x_label)
        
        from PySide6.QtWidgets import QDoubleSpinBox
        self.map_offset_x_spinbox = QDoubleSpinBox()
        self.map_offset_x_spinbox.setRange(-10000.0, 10000.0)
        self.map_offset_x_spinbox.setSingleStep(10.0)
        self.map_offset_x_spinbox.setValue(0.0)
        self.map_offset_x_spinbox.setSuffix(" m")
        self.map_offset_x_spinbox.setToolTip("GPS offset: X in meters (positive = move trajectory east)")
        self.map_offset_x_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                min-width: 80px;
            }
        """)
        self.map_offset_x_spinbox.valueChanged.connect(self.on_map_offset_changed)
        self.map_offset_x_spinbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.map_offset_x_spinbox)
        
        # Y offset
        self.map_offset_y_label = QLabel("Y:")
        self.map_offset_y_label.setStyleSheet("color: #333; font-size: 10px;")
        self.map_offset_y_label.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.map_offset_y_label)
        
        self.map_offset_y_spinbox = QDoubleSpinBox()
        self.map_offset_y_spinbox.setRange(-10000.0, 10000.0)
        self.map_offset_y_spinbox.setSingleStep(10.0)
        self.map_offset_y_spinbox.setValue(0.0)
        self.map_offset_y_spinbox.setSuffix(" m")
        self.map_offset_y_spinbox.setToolTip("GPS offset: Y in meters (positive = move trajectory north)")
        self.map_offset_y_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: #333;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #999;
                border-radius: 3px;
                min-width: 80px;
            }
        """)
        self.map_offset_y_spinbox.valueChanged.connect(self.on_map_offset_changed)
        self.map_offset_y_spinbox.setVisible(False)  # Hidden until map is loaded
        map_header_layout.addWidget(self.map_offset_y_spinbox)
        
        map_header_layout.addSpacing(10)
        
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
        self.clear_route_btn.clicked.connect(lambda: self.clear_route_display(hide_subsections=True))
        route_input_layout.addWidget(self.clear_route_btn)
        
        route_group_layout.addLayout(route_input_layout)
        
        # Route info label
        self.route_info_label = QLabel("")
        self.route_info_label.setStyleSheet("color: #666; font-size: 9px;")
        self.route_info_label.setWordWrap(True)
        route_group_layout.addWidget(self.route_info_label)
        
        # ---- GPS Points Path Checkbox ----
        gps_path_layout = QHBoxLayout()
        gps_path_layout.setSpacing(8)
        
        self.draw_gps_points_path_checkbox = QCheckBox("Draw GPS points path")
        self.draw_gps_points_path_checkbox.setToolTip("Display the blue GPS points and route path on the map")
        self.draw_gps_points_path_checkbox.setChecked(True)  # Enabled by default
        self.draw_gps_points_path_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.draw_gps_points_path_checkbox.stateChanged.connect(self._on_draw_gps_points_path_changed)
        gps_path_layout.addWidget(self.draw_gps_points_path_checkbox)
        gps_path_layout.addStretch()
        route_group_layout.addLayout(gps_path_layout)
        
        # ---- Route Repair Subsection ----
        # Line 1: Fix Invalid Segments checkbox + fix information
        repair_line1_layout = QHBoxLayout()
        repair_line1_layout.setSpacing(8)
        
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
        self.fix_invalid_segments_checkbox.stateChanged.connect(self._on_fix_invalid_segments_changed)
        repair_line1_layout.addWidget(self.fix_invalid_segments_checkbox)
        
        # New number of segments (shown when fix is applied)
        self.new_segments_count_label = QLabel("")
        self.new_segments_count_label.setStyleSheet("color: #333; font-size: 10px; font-weight: bold;")
        repair_line1_layout.addWidget(self.new_segments_count_label)
        
        # Invalid segments fix info label
        self.invalid_segments_info_label = QLabel("")
        self.invalid_segments_info_label.setStyleSheet("color: #333; font-size: 10px;")
        repair_line1_layout.addWidget(self.invalid_segments_info_label)
        
        repair_line1_layout.addStretch()
        route_group_layout.addLayout(repair_line1_layout)
        
        # Line 2: Trim Start/End checkbox + Real Start/Destination labels
        repair_line2_layout = QHBoxLayout()
        repair_line2_layout.setSpacing(8)
        
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
        repair_line2_layout.addWidget(self.fix_route_checkbox)
        
        # Real start point label
        self.real_start_label = QLabel("Real Start: N/A")
        self.real_start_label.setStyleSheet("color: #333; font-size: 10px;")
        repair_line2_layout.addWidget(self.real_start_label)
        
        # Real destination point label
        self.real_destination_label = QLabel("Real Destination: N/A")
        self.real_destination_label.setStyleSheet("color: #333; font-size: 10px;")
        repair_line2_layout.addWidget(self.real_destination_label)
        
        repair_line2_layout.addStretch()
        route_group_layout.addLayout(repair_line2_layout)
        
        # Line 3: Show Polygon checkbox (polygon around trajectory, inner edges in red)
        repair_line3_layout = QHBoxLayout()
        repair_line3_layout.setSpacing(8)
        self.show_polygon_checkbox = QCheckBox("Show Polygon")
        self.show_polygon_checkbox.setToolTip("Draw polygon around trajectory and color edges inside in red")
        self.show_polygon_checkbox.setChecked(True)
        self.show_polygon_checkbox.setStyleSheet("""
            QCheckBox {
                color: #333;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.show_polygon_checkbox.stateChanged.connect(self._on_show_polygon_changed)
        repair_line3_layout.addWidget(self.show_polygon_checkbox)
        repair_line3_layout.addStretch()
        route_group_layout.addLayout(repair_line3_layout)
        
        # ---- Trajectory subsets (one subsection per segment, including single segment) ----
        self.segment_subsections_group = QGroupBox("Trajectory subsets")
        self.segment_subsections_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 10px;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)
        self.segment_subsections_layout = QVBoxLayout()
        self.segment_subsections_layout.setSpacing(6)
        self.segment_subsections_group.setLayout(self.segment_subsections_layout)
        self.segment_subsections_group.setVisible(False)
        self._segment_show_candidate_checkboxes = []
        self._segment_show_sumo_route_checkboxes = []
        self._segment_sumo_route_status_labels = []
        route_group_layout.addWidget(self.segment_subsections_group)
        
        self.route_group.setLayout(route_group_layout)
        self.route_group.setVisible(False)  # Hidden until map and dataset ready
        controls_layout.addWidget(self.route_group)
        
        # ---- Dataset Generation Section ----
        self.dataset_gen_group = QGroupBox("📦 Dataset Generation")
        self.dataset_gen_group.setStyleSheet("""
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
        dataset_gen_layout = QVBoxLayout()
        dataset_gen_layout.setSpacing(8)
        
        # Trajectory range: start number, count, and last number (linked)
        traj_range_layout = QHBoxLayout()
        traj_range_layout.addWidget(QLabel("Start trajectory:"))
        self.dataset_start_traj_spin = QLineEdit()
        self.dataset_start_traj_spin.setPlaceholderText("1")
        self.dataset_start_traj_spin.setFixedWidth(60)
        self.dataset_start_traj_spin.setToolTip("Starting trajectory number (1-based)")
        traj_range_layout.addWidget(self.dataset_start_traj_spin)
        traj_range_layout.addWidget(QLabel("Count:"))
        self.dataset_count_spin = QLineEdit()
        self.dataset_count_spin.setPlaceholderText("10")
        self.dataset_count_spin.setFixedWidth(50)
        self.dataset_count_spin.setToolTip("Number of trajectories to process")
        traj_range_layout.addWidget(self.dataset_count_spin)
        traj_range_layout.addWidget(QLabel("Last:"))
        self.dataset_last_traj_spin = QLineEdit()
        self.dataset_last_traj_spin.setPlaceholderText("10")
        self.dataset_last_traj_spin.setFixedWidth(60)
        self.dataset_last_traj_spin.setToolTip("Last trajectory number to process")
        traj_range_layout.addWidget(self.dataset_last_traj_spin)
        traj_range_layout.addWidget(QLabel("Workers:"))
        self.dataset_workers_spin = QLineEdit()
        self.dataset_workers_spin.setPlaceholderText("4")
        self.dataset_workers_spin.setFixedWidth(40)
        self.dataset_workers_spin.setToolTip("Number of parallel workers (1=single-threaded, 4=faster)")
        traj_range_layout.addWidget(self.dataset_workers_spin)
        traj_range_layout.addStretch()
        dataset_gen_layout.addLayout(traj_range_layout)
        
        # Output folder path
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output folder:"))
        self.dataset_output_path = QLineEdit()
        self.dataset_output_path.setPlaceholderText("Select output folder...")
        self.dataset_output_path.setReadOnly(True)
        output_layout.addWidget(self.dataset_output_path)
        browse_output_btn = QPushButton("📁")
        browse_output_btn.setFixedWidth(36)
        browse_output_btn.setToolTip("Browse for output folder")
        browse_output_btn.clicked.connect(self._browse_dataset_output_folder)
        output_layout.addWidget(browse_output_btn)
        dataset_gen_layout.addLayout(output_layout)
        
        # Start/Stop buttons (hidden until start and output are set)
        self.dataset_buttons_layout = QHBoxLayout()
        self.dataset_start_btn = QPushButton("▶ Start")
        self.dataset_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.dataset_start_btn.clicked.connect(self._on_dataset_start_clicked)
        self.dataset_stop_btn = QPushButton("⏹ Stop")
        self.dataset_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.dataset_stop_btn.clicked.connect(self._on_dataset_stop_clicked)
        self.dataset_start_btn.setVisible(False)
        self.dataset_stop_btn.setVisible(False)
        self.dataset_buttons_layout.addWidget(self.dataset_start_btn)
        self.dataset_buttons_layout.addWidget(self.dataset_stop_btn)
        self.dataset_buttons_layout.addStretch()
        dataset_gen_layout.addLayout(self.dataset_buttons_layout)
        
        # Link count <-> last trajectory (when one changes, update the other)
        def _on_dataset_count_changed():
            txt = self.dataset_count_spin.text().strip()
            if not txt:
                return
            try:
                n = int(txt)
                start_txt = self.dataset_start_traj_spin.text().strip()
                start = int(start_txt) if start_txt else 1
                self.dataset_last_traj_spin.setText(str(start + n - 1))
            except ValueError:
                pass

        def _on_dataset_last_changed():
            txt = self.dataset_last_traj_spin.text().strip()
            if not txt:
                return
            try:
                last = int(txt)
                start_txt = self.dataset_start_traj_spin.text().strip()
                start = int(start_txt) if start_txt else 1
                if last >= start:
                    self.dataset_count_spin.setText(str(last - start + 1))
            except ValueError:
                pass

        def _on_dataset_start_changed():
            txt = self.dataset_start_traj_spin.text().strip()
            if txt:
                _on_dataset_count_changed()

        self.dataset_count_spin.textChanged.connect(_on_dataset_count_changed)
        self.dataset_last_traj_spin.textChanged.connect(_on_dataset_last_changed)
        self.dataset_start_traj_spin.textChanged.connect(_on_dataset_start_changed)
        self.dataset_count_spin.setText("10")  # Set default, triggers last update

        self.dataset_start_traj_spin.textChanged.connect(self._update_dataset_buttons_visibility)
        self.dataset_output_path.textChanged.connect(self._update_dataset_buttons_visibility)
        self.dataset_output_path.textChanged.connect(lambda _: self._schedule_save_settings())

        self.dataset_gen_group.setLayout(dataset_gen_layout)
        self.dataset_gen_group.setVisible(False)  # Hidden until map and dataset ready
        controls_layout.addWidget(self.dataset_gen_group)
        
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
        self._schedule_save_settings()
    
    def on_train_path_changed(self, path: str):
        """Handle train CSV path change."""
        self._schedule_path_validation(path, "train")
        self._schedule_save_settings()
    
    def on_test_path_changed(self, path: str):
        """Handle test CSV path change."""
        self._schedule_path_validation(path, "test")
        self._schedule_save_settings()
    
    def on_zones_changed(self, value: int):
        """Handle zones slider value change."""
        self.zones_value_label.setText(str(value))
        self._schedule_save_settings()
    
    def on_osm_map_changed(self, state: int):
        """Handle OSM map checkbox change."""
        self._schedule_save_settings()
        
        if self.network_parser:
            show_osm_map = self.osm_map_checkbox.isChecked()
            self.log(f"OSM map: {'enabled' if show_osm_map else 'disabled'}")
            
            # Show loading indicator
            if show_osm_map:
                self.osm_map_loading_label.setText("⏳")
                self.osm_map_checkbox.setEnabled(False)
                
                # Connect to the finished signal (disconnect first to avoid duplicate connections)
                try:
                    self.map_view.osm_map_loading_finished.disconnect(self._on_osm_map_loading_finished)
                except RuntimeError:
                    pass  # Not connected yet
                self.map_view.osm_map_loading_finished.connect(self._on_osm_map_loading_finished)
                
                # Use QTimer to defer the work
                QTimer.singleShot(50, self._do_osm_map_toggle)
            else:
                # Disabling is quick, do it directly
                self.map_view.set_osm_map_visible(False)
    
    def _do_osm_map_toggle(self):
        """Actually toggle OSM map visibility."""
        self.map_view.set_osm_map_visible(True)
        # Note: loading indicator is cleared when osm_map_loading_finished signal is emitted
    
    def _on_osm_map_loading_finished(self):
        """Handle OSM map loading completion."""
        self.osm_map_loading_label.setText("")
        self.osm_map_checkbox.setEnabled(True)
        self.log("OSM map loaded")
    
    def on_show_network_changed(self, state: int):
        """Handle show network checkbox change."""
        self._schedule_save_settings()
        
        if self.map_view:
            show_network = self.show_network_checkbox.isChecked()
            self.log(f"SUMO network: {'shown' if show_network else 'hidden'}")
            self.map_view.set_network_visible(show_network)
    
    def on_map_offset_changed(self, value: float):
        """Store GPS offset values. Applied when user clicks Show route."""
        self._schedule_save_settings()
        
        if self.map_view:
            x_offset = self.map_offset_x_spinbox.value()
            y_offset = self.map_offset_y_spinbox.value()
            self.map_view.set_map_offset(x_offset, y_offset)
            self.log(f"GPS offset: X={x_offset:.1f}m, Y={y_offset:.1f}m")
    
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
                self.dataset_gen_group.setVisible(True)
                self._update_dataset_buttons_visibility()
                self.log("✅ Map and dataset ready - Zone and Route configuration enabled")
                # Load trip count from train dataset
                self.load_trip_count()
        else:
            self.zones_group.setVisible(False)
            self.route_group.setVisible(False)
            self.dataset_gen_group.setVisible(False)
    
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
            self.osm_map_checkbox.setVisible(True)  # Show the OSM map checkbox
            self.osm_map_loading_label.setVisible(False)
            self.show_network_checkbox.setVisible(True)  # Show the network checkbox
            self.map_offset_label.setVisible(True)  # Show offset controls
            self.map_offset_x_label.setVisible(True)
            self.map_offset_y_label.setVisible(True)
            self.map_offset_x_spinbox.setVisible(True)
            self.map_offset_y_spinbox.setVisible(True)
            self.load_network_async(net_file)
        elif porto_net_file.exists():
            self.map_status.setText("✅ Network map available (Porto folder)")
            self.map_status.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.download_map_btn.setVisible(False)  # Hide the button when map is available
            self.osm_map_checkbox.setVisible(True)  # Show the OSM map checkbox
            self.osm_map_loading_label.setVisible(False)
            self.show_network_checkbox.setVisible(True)  # Show the network checkbox
            self.map_offset_label.setVisible(True)  # Show offset controls
            self.map_offset_x_label.setVisible(True)
            self.map_offset_y_label.setVisible(True)
            self.map_offset_x_spinbox.setVisible(True)
            self.map_offset_y_spinbox.setVisible(True)
            self.load_network_async(porto_net_file)
        else:
            self.map_status.setText("❌ Network map not found\nClick to download from OpenStreetMap")
            self.map_status.setStyleSheet("color: #f44336; font-size: 11px;")
            self.download_map_btn.setVisible(True)
            self.download_map_btn.setEnabled(True)
            self.osm_map_checkbox.setVisible(False)
            self.osm_map_loading_label.setVisible(False)
            self.show_network_checkbox.setVisible(False)
            self.map_offset_label.setVisible(False)
            self.map_offset_x_label.setVisible(False)
            self.map_offset_y_label.setVisible(False)
            self.map_offset_x_spinbox.setVisible(False)
            self.map_offset_y_spinbox.setVisible(False)
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

    def _update_dataset_buttons_visibility(self):
        """Show Start/Stop buttons when start trajectory and output path are set."""
        start_txt = self.dataset_start_traj_spin.text().strip()
        output_txt = self.dataset_output_path.text().strip()
        show = bool(start_txt and output_txt)
        self.dataset_start_btn.setVisible(show)
        self.dataset_stop_btn.setVisible(show)
        if show and self._dataset_gen_worker and self._dataset_gen_worker.isRunning():
            self.dataset_start_btn.setEnabled(False)
            self.dataset_stop_btn.setEnabled(True)
        elif show:
            self.dataset_start_btn.setEnabled(True)
            self.dataset_stop_btn.setEnabled(False)

    def _browse_dataset_output_folder(self):
        """Open folder dialog to select output directory for dataset generation."""
        current = self.dataset_output_path.text().strip()
        start_dir = current if current and Path(current).exists() else self.project_path
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Dataset",
            start_dir,
            options=QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.dataset_output_path.setText(folder)
            self.log(f"Dataset output folder set to: {folder}")

    def _on_dataset_start_clicked(self):
        """Start dataset generation."""
        train_path = self.train_path_input.text().strip()
        if not train_path or not Path(train_path).exists():
            self.log("❌ train.csv not found - select a valid train dataset first")
            QMessageBox.warning(self, "Dataset Error", "Please select a valid train.csv file first.")
            return
        output_path = self.dataset_output_path.text().strip()
        if not output_path:
            self.log("❌ Select an output folder first")
            QMessageBox.warning(self, "Dataset Error", "Please select an output folder.")
            return
        try:
            start_traj = int(self.dataset_start_traj_spin.text().strip() or "1")
            last_traj = int(self.dataset_last_traj_spin.text().strip() or "10")
        except ValueError:
            self.log("❌ Invalid trajectory numbers")
            QMessageBox.warning(self, "Dataset Error", "Please enter valid start and last trajectory numbers.")
            return
        if start_traj < 1 or last_traj < start_traj:
            self.log("❌ Start must be ≥ 1 and last must be ≥ start")
            QMessageBox.warning(self, "Dataset Error", "Start must be ≥ 1 and last must be ≥ start.")
            return
        if not self.network_parser:
            self.log("❌ Network not loaded")
            QMessageBox.warning(self, "Dataset Error", "Please wait for the network map to load.")
            return
        network_path = getattr(self, 'network_file_path', None) or ''
        if not network_path or not Path(network_path).exists():
            self.log("❌ Network file path not available")
            QMessageBox.warning(self, "Dataset Error", "Network file path not available. Reload the network.")
            return

        try:
            workers = int(self.dataset_workers_spin.text().strip() or "4")
            workers = max(1, min(workers, 32))
        except ValueError:
            workers = 4

        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        if y_min == 0 and y_max == 0:
            self.log("❌ Network bounds not ready")
            QMessageBox.warning(self, "Dataset Error", "Network bounds not ready. Please wait for map to load.")
            return

        offset_x = self.map_offset_x_spinbox.value()
        offset_y = self.map_offset_y_spinbox.value()
        self._dataset_gen_cancelled = False
        self._dataset_gen_worker = DatasetGenerationWorker(
            train_path=train_path,
            output_path=output_path,
            start_traj=start_traj,
            last_traj=last_traj,
            network_path=network_path,
            network_parser=self.network_parser,
            y_min=y_min,
            y_max=y_max,
            workers=workers,
            offset_x=offset_x,
            offset_y=offset_y,
            cancelled_callback=lambda: self._dataset_gen_cancelled,
        )
        self._dataset_gen_worker.progress.connect(self._on_dataset_gen_progress)
        self._dataset_gen_worker.finished.connect(self._on_dataset_gen_finished)
        self.dataset_start_btn.setEnabled(False)
        self.dataset_stop_btn.setEnabled(True)
        self.progress_group.setVisible(True)
        self.progress_group.setTitle("⏳ Dataset Generation")
        self.progress_bar.setMaximum(max(1, last_traj - start_traj + 1))
        self.progress_bar.setValue(0)
        self.progress_status.setText(f"Processing trajectories {start_traj}–{last_traj}...")
        self.log(f"▶ Starting dataset generation: trajectories {start_traj}–{last_traj} → {output_path}")
        self._dataset_gen_worker.start()

    def _on_dataset_stop_clicked(self):
        """Stop dataset generation."""
        self._dataset_gen_cancelled = True
        self.log("⏹ Stop requested...")

    def _on_dataset_gen_progress(self, current: int, total: int, status: str):
        """Handle dataset generation progress."""
        self.progress_bar.setValue(current)
        self.progress_status.setText(status)
        # Log throttled - status already includes progress when throttled
        self.log(f"  {status}")

    def _on_dataset_gen_finished(self, saved_count: int, total_processed: int, error_msg: str):
        """Handle dataset generation completion."""
        self._dataset_gen_worker = None
        self.dataset_start_btn.setEnabled(True)
        self.dataset_stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_group.setTitle("⏳ Download Progress")
        if error_msg:
            self.progress_status.setText(f"Stopped: {error_msg}")
            self.log(f"Dataset generation stopped: {error_msg}")
        else:
            self.progress_status.setText(f"Complete: {saved_count} saved from {total_processed} processed")
            self.log(f"✅ Dataset generation complete: {saved_count} trajectory file(s) saved from {total_processed} processed")
    
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
        """Load network asynchronously and store the file path."""
        self.network_file_path = str(net_file)  # Store network file path for TraCI routing
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
            
            # Build spatial index for fast edge lookups (Step 0)
            # Include ALL edges in spatial index for candidate finding
            # (We need to find the closest edges, even if they don't allow passengers,
            #  because the closest passenger-allowed edge might be far away)
            try:
                self.log("Building spatial index for edge lookups...")
                all_edges = self.network_parser.get_edges()
                # Include ALL edges in spatial index for accurate candidate finding
                # We'll filter to passenger-allowed edges later when calculating routes
                self._edge_spatial_index = EdgeSpatialIndex(all_edges, cell_size=500.0)
                self.log(f"✓ Spatial index built: {len(all_edges)} edges indexed (all edges for accurate candidate finding)")
            except Exception as e:
                self.log(f"⚠️ Failed to build spatial index: {e}")
                self._edge_spatial_index = None
            
            # Load sumolib network once for routing (heavy operation, do it once)
            if self.network_file_path:
                try:
                    import sumolib
                    self.log(f"Loading SUMO network for routing from: {self.network_file_path}")
                    self.sumo_net = sumolib.net.readNet(self.network_file_path)
                    edge_count = len(self.sumo_net.getEdges())
                    node_count = len(self.sumo_net.getNodes())
                    self.log(f"✓ SUMO network loaded for routing: {edge_count} edges, {node_count} nodes")
                except ImportError:
                    self.log("⚠️ sumolib not available. SUMO route overlay will not work.")
                    self.sumo_net = None
                except Exception as e:
                    self.log(f"⚠️ Failed to load SUMO network for routing: {e}")
                    self.sumo_net = None
            
            # Load network
            self.map_view.load_network(self.network_parser)
            
            # Zoom to Porto city center area
            self._zoom_to_porto_center()
            
            # Get network statistics - load ALL edges from XML (no filtering)
            all_edges = self.network_parser.get_edges()
            junctions = self.network_parser.get_junctions()
            bounds = self.network_parser.get_bounds()
            
            # Count passenger-allowed edges for info (but don't filter)
            roads = {eid: edata for eid, edata in all_edges.items() if edata.get('allows_passenger', True)}
            
            self.map_status_label.setText(
                f"Map loaded: {len(all_edges)} edges ({len(roads)} passenger-allowed), {len(junctions)} junctions"
            )
            self.log(f"Network loaded successfully: {len(all_edges)} total edges (including all types: passenger, pedestrian, bicycle, etc.), {len(junctions)} junctions")
            
            # Apply OSM map setting if checked
            if self.osm_map_checkbox.isChecked():
                self.map_view.set_osm_map_visible(True)
            
            # Apply network visibility setting
            self.map_view.set_network_visible(self.show_network_checkbox.isChecked())
            
            # Apply map offset setting
            self.map_view.set_map_offset(self.map_offset_x_spinbox.value(), self.map_offset_y_spinbox.value())
            
            self.map_status_label.setStyleSheet("color: #333; font-size: 12px;")
            
            if bounds:
                self.log(f"Bounds: ({bounds['x_min']:.1f}, {bounds['y_min']:.1f}) to ({bounds['x_max']:.1f}, {bounds['y_max']:.1f})")
            
            # Show map view
            self.map_stack.setCurrentIndex(0)
            
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
                self.osm_map_checkbox.setVisible(True)  # Show the OSM map checkbox
                self.osm_map_loading_label.setVisible(False)
                self.show_network_checkbox.setVisible(True)  # Show the network checkbox
                self.map_offset_label.setVisible(True)  # Show offset controls
                self.map_offset_x_label.setVisible(True)
                self.map_offset_y_label.setVisible(True)
                self.map_offset_x_spinbox.setVisible(True)
                self.map_offset_y_spinbox.setVisible(True)
                self.load_network_async(net_file)
            
            # Check dataset status
            self._check_dataset_status()
        else:
            self.progress_bar.setValue(0)
            QMessageBox.warning(self, "Download Failed", message)
            self.download_map_btn.setVisible(True)
            self.download_map_btn.setEnabled(True)
            self.osm_map_checkbox.setVisible(False)
            self.osm_map_loading_label.setVisible(False)
            self.show_network_checkbox.setVisible(False)
            self.map_offset_label.setVisible(False)
            self.map_offset_x_label.setVisible(False)
            self.map_offset_y_label.setVisible(False)
            self.map_offset_x_spinbox.setVisible(False)
            self.map_offset_y_spinbox.setVisible(False)
        
        # Hide progress section after a short delay
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
    
    def _schedule_save_settings(self, delay_ms: int = 400):
        """Schedule a debounced save; cancels any pending save."""
        self._save_settings_timer.stop()
        self._save_settings_timer.start(delay_ms)
    
    def _schedule_path_validation(self, path: str, file_type: str, delay_ms: int = 300):
        """Schedule debounced path validation."""
        self._validation_args = (path, file_type)
        self._validation_timer.stop()
        self._validation_timer.start(delay_ms)
    
    def _on_validation_timer_fired(self):
        """Run deferred path validation and check zones."""
        if self._validation_args:
            path, file_type = self._validation_args
            self._validation_args = None
            self.validate_dataset_path(path, file_type)
            self.check_zones_visibility()
    
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
            'show_osm_map': self.osm_map_checkbox.isChecked(),
            'show_network': self.show_network_checkbox.isChecked(),
            'map_offset_x': self.map_offset_x_spinbox.value(),
            'map_offset_y': self.map_offset_y_spinbox.value(),
            'fix_route': self.fix_route_checkbox.isChecked(),
            'fix_invalid_segments': self.fix_invalid_segments_checkbox.isChecked(),
            'show_polygon': self.show_polygon_checkbox.isChecked(),
            'draw_gps_points_path': self.draw_gps_points_path_checkbox.isChecked(),
            'dataset_output_folder': self.dataset_output_path.text().strip(),
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
                    self.train_path_input.blockSignals(True)
                    self.train_path_input.setText(settings['train_csv_path'])
                    self.train_path_input.blockSignals(False)
                    self.validate_dataset_path(settings['train_csv_path'], "train")
                
                if 'test_csv_path' in settings and settings['test_csv_path']:
                    self.test_path_input.blockSignals(True)
                    self.test_path_input.setText(settings['test_csv_path'])
                    self.test_path_input.blockSignals(False)
                    self.validate_dataset_path(settings['test_csv_path'], "test")
                
                if 'num_zones' in settings:
                    self.zones_slider.setValue(settings['num_zones'])
                    self.zones_value_label.setText(str(settings['num_zones']))
                
                if 'train_trip_count' in settings and settings['train_trip_count']:
                    self._train_trip_count = settings['train_trip_count']
                
                if 'show_osm_map' in settings:
                    self.osm_map_checkbox.setChecked(settings['show_osm_map'])
                
                if 'show_network' in settings:
                    self.show_network_checkbox.setChecked(settings['show_network'])
                    # Apply network visibility if map is already loaded
                    if self.map_view:
                        self.map_view.set_network_visible(settings['show_network'])
                
                # Block signals during settings loading to avoid duplicate logs
                if 'map_offset_x' in settings or 'map_offset_y' in settings:
                    # Temporarily block signals to prevent duplicate log messages
                    self.map_offset_x_spinbox.blockSignals(True)
                    self.map_offset_y_spinbox.blockSignals(True)
                    
                    if 'map_offset_x' in settings:
                        self.map_offset_x_spinbox.setValue(settings['map_offset_x'])
                    
                    if 'map_offset_y' in settings:
                        self.map_offset_y_spinbox.setValue(settings['map_offset_y'])
                    
                    # Apply both offsets at once if map_view exists
                    if self.map_view:
                        x_offset = self.map_offset_x_spinbox.value()
                        y_offset = self.map_offset_y_spinbox.value()
                        self.map_view.set_map_offset(x_offset, y_offset)
                        # Log once with correct values
                        self.log(f"Map offset: X={x_offset:.1f}m, Y={y_offset:.1f}m")
                    
                    # Re-enable signals
                    self.map_offset_x_spinbox.blockSignals(False)
                    self.map_offset_y_spinbox.blockSignals(False)
                
                if 'fix_route' in settings:
                    self.fix_route_checkbox.setChecked(settings['fix_route'])
                
                if 'fix_invalid_segments' in settings:
                    self.fix_invalid_segments_checkbox.setChecked(settings['fix_invalid_segments'])
                
                if 'show_polygon' in settings:
                    self.show_polygon_checkbox.setChecked(settings['show_polygon'])
                
                if 'draw_gps_points_path' in settings:
                    self.draw_gps_points_path_checkbox.setChecked(settings['draw_gps_points_path'])
                
                if 'dataset_output_folder' in settings and settings['dataset_output_folder']:
                    self.dataset_output_path.blockSignals(True)
                    self.dataset_output_path.setText(settings['dataset_output_folder'])
                    self.dataset_output_path.blockSignals(False)
                
                self.check_zones_visibility()
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
    
    def _on_fix_invalid_segments_changed(self):
        """Handle fix invalid segments checkbox change."""
        self._schedule_save_settings()
        # Refresh the route display if a route is currently shown
        if hasattr(self, '_route_items') and self._route_items:
            self.show_selected_route()
    
    def _on_fix_route_changed(self):
        """Handle fix route checkbox change."""
        self._schedule_save_settings()
        # Refresh the route display if a route is currently shown
        if hasattr(self, '_route_items') and self._route_items:
            self.show_selected_route()

    def _on_show_polygon_changed(self):
        """Handle Show Polygon checkbox change."""
        self._schedule_save_settings()
        if hasattr(self, '_route_items') and self._route_items:
            self.show_selected_route()

    def _update_segment_subsections_ui(self, segments: List[List[List[float]]]) -> None:
        """Update trajectory subsets UI: show one subsection per segment (including single segment)."""
        if len(segments) < 1:
            self.segment_subsections_group.setVisible(False)
            # Clear widgets when no segments
            while self.segment_subsections_layout.count():
                item = self.segment_subsections_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._segment_show_candidate_checkboxes.clear()
            self._segment_show_sumo_route_checkboxes.clear()
            self._segment_sumo_route_status_labels.clear()
            return
        # Recreate widgets when segment count changed
        if len(self._segment_show_candidate_checkboxes) != len(segments):
            while self.segment_subsections_layout.count():
                item = self.segment_subsections_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._segment_show_candidate_checkboxes.clear()
            self._segment_show_sumo_route_checkboxes.clear()
            self._segment_sumo_route_status_labels.clear()
            for i, segment in enumerate(segments):
                subsection = QFrame()
                subsection.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 4px; padding: 6px; }")
                sub_layout = QVBoxLayout()
                sub_layout.setSpacing(4)
                sub_layout.setContentsMargins(8, 6, 8, 6)
                seg_label = QLabel(f"Segment {i + 1}")
                seg_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #333;")
                sub_layout.addWidget(seg_label)
                sumo_row = QHBoxLayout()
                cb_sumo = QCheckBox("Show SUMO route")
                cb_sumo.setToolTip("Display the SUMO shortest path (purple) for this subset (view_network logic)")
                cb_sumo.setStyleSheet("font-size: 9px; color: #333;")
                has_route, status_msg = self._segment_has_sumo_route(segment, seg_idx=i)
                if has_route:
                    cb_sumo.setChecked(True)
                    cb_sumo.setEnabled(True)
                    status_label = QLabel("")
                else:
                    cb_sumo.setChecked(False)
                    cb_sumo.setEnabled(False)
                    status_label = QLabel(status_msg or "No route found")
                    status_label.setStyleSheet("font-size: 9px; color: #c62828;")
                status_label.setMinimumWidth(200)
                status_label.setWordWrap(True)
                cb_sumo.stateChanged.connect(self._on_segment_checkbox_changed)
                sumo_row.addWidget(cb_sumo)
                sumo_row.addWidget(status_label)
                sumo_row.addStretch()
                sub_layout.addLayout(sumo_row)
                self._segment_show_sumo_route_checkboxes.append(cb_sumo)
                self._segment_sumo_route_status_labels.append(status_label)
                cb_candidate = QCheckBox("Show candidate edges (green/orange)")
                cb_candidate.setToolTip("Display the orange (closest to start/end) and green (top matching) edges for this subset")
                cb_candidate.setStyleSheet("font-size: 9px; color: #333;")
                cb_candidate.setChecked(False)
                cb_candidate.stateChanged.connect(self._on_segment_checkbox_changed)
                sub_layout.addWidget(cb_candidate)
                self._segment_show_candidate_checkboxes.append(cb_candidate)
                subsection.setLayout(sub_layout)
                self.segment_subsections_layout.addWidget(subsection)
        else:
            # Same segment count: update status labels for new route data
            for i, segment in enumerate(segments):
                if i >= len(self._segment_sumo_route_status_labels):
                    break
                has_route, status_msg = self._segment_has_sumo_route(segment, seg_idx=i)
                cb_sumo = self._segment_show_sumo_route_checkboxes[i]
                status_label = self._segment_sumo_route_status_labels[i]
                cb_sumo.blockSignals(True)
                if has_route:
                    cb_sumo.setChecked(True)
                    cb_sumo.setEnabled(True)
                    status_label.setText("")
                else:
                    cb_sumo.setChecked(False)
                    cb_sumo.setEnabled(False)
                    status_label.setText(status_msg or "No route found")
                    status_label.setStyleSheet("font-size: 9px; color: #c62828;")
                cb_sumo.blockSignals(False)
        self.segment_subsections_group.setVisible(True)

    def _on_segment_checkbox_changed(self):
        """When segment 'Show candidate edges' or 'Show SUMO route' checkbox changes, update only overlay layers (fast path)."""
        if not self._current_segments:
            return
        # Remove candidate and sumo route items
        for item in self._candidate_edge_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                pass
        self._candidate_edge_items = []
        for item in self._sumo_route_items:
            try:
                self.map_view.scene.removeItem(item)
            except RuntimeError:
                pass
        self._sumo_route_items = []
        # Redraw for checked segments
        for i, segment in enumerate(self._current_segments):
            if i >= len(self._segment_show_candidate_checkboxes) or i >= len(self._segment_show_sumo_route_checkboxes):
                continue
            if self._segment_show_candidate_checkboxes[i].isChecked():
                self._draw_segment_candidate_edges(segment)
            if self._segment_show_sumo_route_checkboxes[i].isChecked():
                self._draw_segment_sumo_route(segment, seg_idx=i)
    
    def _prepare_route_data(self, original_polyline: List[List[float]]):
        """
        Prepare route data by analyzing the polyline once.
        This function performs expensive operations only once and returns all needed data.
        
        Args:
            original_polyline: The original GPS polyline (list of [lon, lat] points)
        
        Returns:
            tuple: (
                invalid_segments: TripValidationResult object with invalid segment info,
                real_start_idx: int - real start index for entire original GPS segment,
                real_end_idx: int - real end index for entire original GPS segment,
                segment_trim_data: List[Tuple[int, int]] - list of (real_start, real_last) 
                    for each split GPS segment (empty list if no invalid segments exist)
            )
        """
        # Validate original polyline to find invalid segments
        validation_result = validate_trip_segments(original_polyline)
        
        # Detect real start and end points for the entire original GPS segment
        real_start_idx, real_end_idx = self._detect_real_start_and_end(original_polyline)
        
        # Calculate segment trim data (real_start, real_last) for each split segment
        segment_trim_data = []
        if validation_result.invalid_segment_count > 0:
            # Split at invalid segments to get individual segments
            split_segments = self._split_at_invalid_segments(original_polyline)
            
            # For each split segment, calculate its real start and end
            for segment in split_segments:
                if len(segment) >= 2:
                    seg_start, seg_end = self._detect_real_start_and_end(segment)
                    segment_trim_data.append((seg_start, seg_end))
        # If no invalid segments, segment_trim_data remains empty list
        
        return validation_result, real_start_idx, real_end_idx, segment_trim_data
    
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
        
        # Check if we have cached data for this route number
        if self._cached_route_num == route_num and self._cached_route_data is not None:
            # Use cached data (checkbox change - no need to reload)
            original_polyline = self._cached_original_polyline
            invalid_segments, real_start_idx, real_end_idx, segment_trim_data = self._cached_route_data
            self.log(f"Using cached data for route #{route_num}")
            self._apply_route_display(
                original_polyline, invalid_segments, real_start_idx, real_end_idx, segment_trim_data, route_num
            )
            return
        
        # New route - load in background
        if self._route_load_worker and self._route_load_worker.isRunning():
            return  # Already loading
        self.log(f"Loading route #{route_num}...")
        self.route_info_label.setText(f"Loading route #{route_num}...")
        self.show_route_btn.setEnabled(False)
        if not hasattr(self, '_last_route_num') or self._last_route_num != route_num:
            self._step1_route_edges = set()
            self._step1_final_segments = []
            self._step2_coord_logged = False
            self._step2_comparison_logged = False
            self._step2_distance_diagnostic_logged = False
            self._step2_return_logged = False
            self._step2_missing_diagnostic_logged = False
        self._last_route_num = route_num
        self._route_load_worker = RouteLoadWorker(train_path, route_num)
        self._route_load_worker.finished.connect(self._on_route_load_finished)
        self._route_load_worker.start()

    def _on_route_load_finished(self, success: bool, result, error_msg: str):
        """Handle RouteLoadWorker completion."""
        self.show_route_btn.setEnabled(True)
        self._route_load_worker = None
        route_num = self.route_spinbox.value()
        if not success:
            self.route_info_label.setText(
                f"Route #{route_num}: {error_msg[:40]}" if error_msg else "Error loading route"
            )
            self.route_info_label.setStyleSheet("color: #f44336; font-size: 9px;")
            self.log(f"Error loading route #{route_num}: {error_msg}")
            self._cached_route_num = None
            self._cached_original_polyline = None
            self._cached_route_data = None
            return
        original_polyline, invalid_segments, real_start_idx, real_end_idx, segment_trim_data = result
        self._cached_route_num = route_num
        self._cached_original_polyline = original_polyline
        self._cached_route_data = (invalid_segments, real_start_idx, real_end_idx, segment_trim_data)
        self._apply_route_display(
            original_polyline, invalid_segments, real_start_idx, real_end_idx, segment_trim_data, route_num
        )

    def _apply_route_display(
        self, original_polyline, invalid_segments, real_start_idx, real_end_idx, segment_trim_data, route_num
    ):
        """Apply route data and draw on map (must run on main thread)."""
        original_length = len(original_polyline)
        try:
            # Update labels with real start/end info
            if real_start_idx < len(original_polyline):
                self.real_start_label.setText(f"Real Start: Point {real_start_idx + 1}")
            else:
                self.real_start_label.setText("Real Start: N/A")
            
            if real_end_idx < len(original_polyline):
                self.real_destination_label.setText(f"Real Destination: Point {real_end_idx + 1}")
            else:
                self.real_destination_label.setText("Real Destination: N/A")
            
            # Determine what to display based on checkbox states
            fix_invalid = self.fix_invalid_segments_checkbox.isChecked()
            trim_start_end = self.fix_route_checkbox.isChecked()
            
            # Build segments to display
            if fix_invalid and invalid_segments.invalid_segment_count > 0:
                # Use split segments
                segments = self._split_at_invalid_segments(original_polyline)
                
                # Apply trim start/end to each segment if trim checkbox is checked
                if trim_start_end and segment_trim_data:
                    trimmed_segments = []
                    for i, segment in enumerate(segments):
                        if i < len(segment_trim_data) and len(segment) >= 2:
                            seg_start, seg_end = segment_trim_data[i]
                            trimmed_seg = segment[seg_start:seg_end + 1]
                            if len(trimmed_seg) >= 2:  # Only add if has at least 2 points
                                trimmed_segments.append(trimmed_seg)
                    segments = trimmed_segments if trimmed_segments else segments
            else:
                # Use original polyline (single segment)
                if trim_start_end:
                    # Apply trim to original
                    segments = [original_polyline[real_start_idx:real_end_idx + 1]]
                else:
                    segments = [original_polyline]
            
            # Store current segments for SUMO route mapping
            self._current_segments = segments
            
            # Update segment subsections UI (visible when 2+ segments)
            self._update_segment_subsections_ui(segments)
            
            # Clear previous route
            self.clear_route_display()
            
            # Update new segments count and invalid segments info label
            if fix_invalid and len(segments) > 0:
                n = len(segments)
                self.new_segments_count_label.setText(f"{n} segment" + ("s" if n != 1 else ""))
            else:
                self.new_segments_count_label.setText("")
            if fix_invalid and invalid_segments.invalid_segment_count > 0:
                invalid_count = invalid_segments.invalid_segment_count
                if len(segments) > 1:
                    total_points = sum(len(seg) for seg in segments)
                    self.invalid_segments_info_label.setText(
                        f"Invalid segments: {invalid_count} | New routes: {len(segments)} segments ({total_points} total points)"
                    )
                else:
                    self.invalid_segments_info_label.setText(f"Invalid segments: {invalid_count} | No split needed")
            elif not fix_invalid and invalid_segments.invalid_segment_count > 0:
                # Show invalid segments info when not fixing
                invalid_count = invalid_segments.invalid_segment_count
                self.invalid_segments_info_label.setText(f"Invalid segments: {invalid_count} (shown in red)")
            else:
                self.invalid_segments_info_label.setText("")
            
            # Draw route on map (only if GPS points path checkbox is checked)
            draw_gps_path = self.draw_gps_points_path_checkbox.isChecked()
            validation_result = None
            if draw_gps_path:
                # Show invalid in red only if fix_invalid is NOT checked
                show_invalid_in_red = not fix_invalid
                validation_result = self._draw_route_on_map(
                    segments, 
                    route_num, 
                    show_invalid_in_red=show_invalid_in_red,
                    original_polyline=original_polyline if show_invalid_in_red else None
                )
                # Draw polygon around trajectory and red inner edges if Show Polygon is checked
                if self.show_polygon_checkbox.isChecked():
                    self._draw_route_polygon_and_red_edges(segments, route_num)
                # Draw candidate edges and SUMO route for each segment subset if checkbox checked
                if self._segment_show_candidate_checkboxes and self._segment_show_sumo_route_checkboxes:
                    for i, segment in enumerate(segments):
                        if i < len(self._segment_show_candidate_checkboxes) and self._segment_show_candidate_checkboxes[i].isChecked():
                            self._draw_segment_candidate_edges(segment)
                        if i < len(self._segment_show_sumo_route_checkboxes) and self._segment_show_sumo_route_checkboxes[i].isChecked():
                            self._draw_segment_sumo_route(segment, seg_idx=i)
            
            # Build route info text
            display_polyline = segments[0] if segments else []
            repair_info = ""
            if trim_start_end:
                repair_info = f" (Trimmed: {original_length} → {len(display_polyline)} points)"
            else:
                repair_info = f" (Original: {original_length} points)"
            
            if validation_result and not validation_result.is_valid and not fix_invalid:
                self.route_info_label.setText(
                    f"Route #{route_num}: {len(display_polyline)} points{repair_info} | "
                    f"⚠️ {validation_result.invalid_segment_count} invalid segment(s)"
                )
                self.route_info_label.setStyleSheet("color: #f44336; font-size: 9px; font-weight: bold;")
            else:
                self.route_info_label.setText(
                    f"Route #{route_num}: {len(display_polyline)} GPS points{repair_info} ✅"
                )
                self.route_info_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-weight: bold;")
            
            log_msg = f"Route #{route_num} displayed"
            if fix_invalid:
                log_msg += f" as {len(segments)} segment(s)"
            else:
                log_msg += f" with {len(display_polyline)} GPS points"
            if trim_start_end:
                log_msg += f" (trimmed from {original_length})"
            self.log(log_msg)
            
        except Exception as e:
            self.route_info_label.setText(f"Error displaying route: {str(e)[:30]}")
            self.route_info_label.setStyleSheet("color: #f44336; font-size: 9px;")
            self.log(f"Error displaying route #{route_num}: {e}")
    
    def _draw_sumo_route_overlay_placeholder(
        self,
        segments: List[List[List[float]]],
        invalid_segments,
        real_start_idx: int,
        real_end_idx: int,
        segment_trim_data: List[Tuple[int, int]],
        sumo_net
    ):
        """
        Draw SUMO route overlay using Step 1 of the algorithm.
        
        Args:
            segments: List of polyline segments to draw routes for
            invalid_segments: TripValidationResult with invalid segment info
            real_start_idx: Real start index for entire original GPS segment
            real_end_idx: Real end index for entire original GPS segment
            segment_trim_data: List of (real_start, real_last) tuples for each split segment
            sumo_net: SUMO network object
        """
        if not sumo_net:
            self.log("⚠️ SUMO network not available for route calculation")
            return
        
        if not self._edge_spatial_index:
            self.log("⚠️ Spatial index not available for route calculation")
            return
        
        if not self.network_parser:
            self.log("⚠️ Network parser not available")
            return
        
        # Process each segment separately
        all_route_edges = []
        final_segments = []  # Store final segments (may be shortened) for star drawing
        # Store final_segments as instance variable for Step 2 to use
        self._step1_final_segments = []
        for seg_idx, segment in enumerate(segments):
            if not segment or len(segment) < 2:
                continue
            
            # Step 1: Calculate initial SUMO route for this segment (with iterative shortening)
            routes, start_candidates, dest_candidates, route_scores, final_segment = self._calculate_step1_route(segment, sumo_net, seg_idx)
            
            # Draw only Rank 1 route (best match) - the selected route
            if routes:
                # Only draw the first route (Rank 1 - best similarity score)
                best_route = routes[0]
                
                # Store Step 1 route edges for comparison with Step 2
                if not hasattr(self, '_step1_route_edges'):
                    self._step1_route_edges = set()
                # Store base edge IDs (without lane #) for comparison
                for edge_id in best_route:
                    base_id = edge_id.split('#')[0]
                    self._step1_route_edges.add(base_id)
                
                # Log Step 1 route edges for debugging
                if seg_idx == 0:  # Only log once per route load
                    self.log(f"📍 Step 1 route edges ({len(best_route)} edges): {best_route[:10]}{'...' if len(best_route) > 10 else ''}")
                    # Log all Step 1 route base IDs
                    step1_base_ids = sorted(self._step1_route_edges)
                    self.log(f"📍 Step 1 route base IDs ({len(step1_base_ids)}): {step1_base_ids}")
                
                route_color = QColor(255, 0, 255)  # Magenta for Rank 1
                
                # Log similarity score of displayed route (Rank 1)
                if route_scores and len(route_scores) > 0:
                    best_score = route_scores[0]
                    self.log(f"📍 Segment {seg_idx + 1}: Displayed SUMO route similarity score: {best_score:.4f} (Rank 1 - Best Match)")
                
                # Draw only Rank 1 route
                all_route_edges.extend(best_route)
                self._draw_route_edges(best_route, color=route_color)
                
                # Log all route scores for reference (but only display Rank 1)
                if route_scores:
                    for route_idx, score in enumerate(route_scores):
                        rank = route_idx + 1
                        if route_idx < len(routes):
                            edges_count = len(routes[route_idx])
                            status = " (displayed)" if rank == 1 else " (not displayed)"
                            self.log(f"  Route Rank {rank}: similarity={score:.4f}, edges={edges_count}{status}")
            
            # Store final segment for star drawing (may be shortened)
            if final_segment:
                final_segments.append(final_segment)
                self._step1_final_segments.append(final_segment)
        
        # Draw stars at GPS points using SUMO coordinates - only for final segments (may be shortened)
        if final_segments:
            self._draw_gps_points_as_stars(final_segments)
        else:
            # Fallback to original segments if no final segments
            self._draw_gps_points_as_stars(segments)
        
        if all_route_edges:
            self.log(f"✅ Step 1: Calculated SUMO route with {len(all_route_edges)} edges across {len(segments)} segment(s)")
        else:
            self.log("⚠️ Step 1: No valid routes calculated")
    
    def _find_candidate_edges(self, lon: float, lat: float, max_candidates: int = None, max_radius: float = None) -> List[Tuple[str, float]]:
        """
        Step 1.1: Find candidate edges for a GPS point.
        
        Args:
            lon: Longitude
            lat: Latitude
            max_candidates: Maximum number of candidates to return (default: INTERMEDIATE_CANDIDATES for intermediate points)
                          If max_radius is specified, this is ignored and all edges within radius are returned
            max_radius: Maximum radius in meters to search for edges. If specified, returns all edges within this radius
                       (used for start/end points). If None, uses max_candidates limit.
        
        Returns:
            List of (edge_id, distance) tuples, sorted by distance (closest first)
        """
        # Use default for intermediate points if not specified
        if max_candidates is None and max_radius is None:
            max_candidates = getattr(self, 'INTERMEDIATE_CANDIDATES', 6)
        
        # Convert GPS to SUMO coordinates
        sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
        if not sumo_coords:
            return []
        
        x, y = sumo_coords
        
        # Determine search radius
        # If max_radius is specified, use it; otherwise use a larger radius for initial search
        search_radius = max_radius if max_radius is not None else 500.0
        
        # Use spatial index to get candidate edges
        candidate_edge_ids = self._edge_spatial_index.get_candidates_in_radius(x, y, radius=search_radius)
        
        if not candidate_edge_ids:
            return []
        
        # Calculate precise distance from GPS point to each candidate edge
        edges_dict = self.network_parser.get_edges()
        edge_distances = []  # List of (edge_id, distance)
        
        for edge_id in candidate_edge_ids:
            if edge_id not in edges_dict:
                continue
            
            edge_data = edges_dict[edge_id]
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            # Use first lane's shape
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Calculate minimum distance from point to edge shape
            min_dist = float('inf')
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                # Calculate distance manually (point to segment)
                dx = x2 - x1
                dy = y2 - y1
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                else:
                    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                    closest_x = x1 + t * dx
                    closest_y = y1 + t * dy
                    dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                min_dist = min(min_dist, dist)
            
            # If max_radius is specified, filter by distance; otherwise include all
            if max_radius is not None:
                if min_dist <= max_radius:
                    edge_distances.append((edge_id, min_dist))
            else:
                if min_dist < float('inf'):
                    edge_distances.append((edge_id, min_dist))
        
        # Deduplicate by base edge ID (ignore lane #)
        # Keep closest variant per base edge
        base_edge_map = {}  # base_id -> (edge_id, distance) - keep closest variant
        for edge_id, distance in edge_distances:
            # Get base edge ID (strip # suffix)
            base_id = edge_id.split('#')[0] if '#' in edge_id else edge_id
            
            # Keep only closest variant per base edge
            if base_id not in base_edge_map or distance < base_edge_map[base_id][1]:
                base_edge_map[base_id] = (edge_id, distance)
        
        # Sort by distance
        candidates = sorted(base_edge_map.values(), key=lambda x: x[1])
        
        # If max_radius was specified, return all candidates within radius
        # Otherwise, limit to max_candidates
        if max_radius is not None:
            return candidates  # Return all within radius
        else:
            return candidates[:max_candidates]  # Limit to max_candidates
    
    def _calculate_route_similarity_score(
        self,
        route_edges: List[str],
        gps_points: List[List[float]],
        seg_idx: int
    ) -> float:
        """
        Calculate similarity score for a route based on how well it matches GPS trajectory.
        
        **Important**: Only GPS points in the provided `gps_points` list are used for scoring.
        This ensures that if the segment was shortened (e.g., points 0 and 21 removed),
        only the GPS points actually used for route calculation (e.g., points 1-20) are
        considered in the similarity score.
        
        Uses Option 2: Weighted Point-to-Edge Distance with Edge Coverage
        - Coverage score: How many GPS points have their candidate edges in the route
        - Distance score: Average distance from GPS points to route edges
        
        Args:
            route_edges: List of edge IDs in the route
            gps_points: List of [lon, lat] GPS points (should be the segment actually used for route calculation)
            seg_idx: Segment index for logging
        
        Returns:
            Similarity score (0-1, higher is better match)
        """
        if not route_edges or not gps_points:
            return 0.0
        
        # Get base edge IDs from route (ignore lane #)
        route_base_ids = {edge_id.split('#')[0] for edge_id in route_edges}
        
        # Get edges dict for distance calculation
        edges_dict = self.network_parser.get_edges()
        
        matched_points = 0
        total_distance = 0.0
        valid_points = 0
        
        # Iterate only over GPS points in the provided segment (not original full segment)
        # This ensures that if points were removed during iterative shortening, they are not included in the score
        for lon, lat in gps_points:
            # Find 3 closest candidate edges for this GPS point
            candidates = self._find_candidate_edges(lon, lat, max_candidates=3)
            
            if not candidates:
                continue
            
            valid_points += 1
            
            # Check if any candidate's base ID is in route (coverage metric)
            candidate_base_ids = {edge_id.split('#')[0] for edge_id, _ in candidates}
            if route_base_ids.intersection(candidate_base_ids):
                matched_points += 1
            
            # Calculate minimum distance from GPS point to any edge in route (distance metric)
            # Convert GPS to SUMO coordinates
            sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if not sumo_coords:
                continue
            
            x, y = sumo_coords
            min_dist_to_route = float('inf')
            
            # Calculate distance to each edge in route
            for edge_id in route_edges:
                if edge_id not in edges_dict:
                    continue
                
                edge_data = edges_dict[edge_id]
                lanes = edge_data.get('lanes', [])
                if not lanes:
                    continue
                
                # Use first lane's shape
                shape = lanes[0].get('shape', [])
                if len(shape) < 2:
                    continue
                
                # Calculate minimum distance from point to edge shape segments
                for i in range(len(shape) - 1):
                    x1, y1 = shape[i]
                    x2, y2 = shape[i + 1]
                    
                    # Calculate distance from point to line segment
                    dx = x2 - x1
                    dy = y2 - y1
                    len_sq = dx * dx + dy * dy
                    
                    if len_sq == 0:
                        dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                    else:
                        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                        closest_x = x1 + t * dx
                        closest_y = y1 + t * dy
                        dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                    
                    min_dist_to_route = min(min_dist_to_route, dist)
            
            if min_dist_to_route < float('inf'):
                total_distance += min_dist_to_route
        
        if valid_points == 0:
            return 0.0
        
        # Calculate coverage score (0-1)
        coverage_score = matched_points / valid_points if valid_points > 0 else 0.0
        
        # Calculate distance score (0-1, using exponential decay)
        avg_distance = total_distance / valid_points if valid_points > 0 else float('inf')
        scale_factor = 100.0  # meters - tune based on GPS accuracy
        distance_score = math.exp(-avg_distance / scale_factor) if avg_distance < float('inf') else 0.0
        
        # Weighted combination (60% coverage, 40% distance)
        coverage_weight = 0.6
        distance_weight = 0.4
        final_score = coverage_weight * coverage_score + distance_weight * distance_score
        
        return final_score
    
    def _calculate_step1_route_single(self, segment: List[List[float]], sumo_net, seg_idx: int) -> Tuple[List[str], List[Tuple[str, float]], List[Tuple[str, float]], List[float], float]:
        """
        Calculate Step 1 route for a single segment (without iterative shortening).
        
        **Important**: The similarity score is calculated using only the GPS points in the provided
        `segment` parameter. If the segment was shortened (e.g., points 0 and 21 removed),
        only the GPS points in this segment (e.g., points 1-20) are used for similarity scoring.
        
        Args:
            segment: List of [lon, lat] GPS points (may be shortened from original)
            sumo_net: SUMO network object
            seg_idx: Segment index for logging
        
        Returns:
            Tuple of (route_edges, start_candidates, dest_candidates, route_scores, best_score):
            - route_edges: List of edge IDs representing the route, or empty list if no route found
            - start_candidates: List of (edge_id, distance) tuples for start point
            - dest_candidates: List of (edge_id, distance) tuples for destination point
            - route_scores: List of similarity scores for each route (sorted by score, best first)
            - best_score: Similarity score of the best route (Rank 1), or 0.0 if no route found
        """
        if len(segment) < 2:
            return [], [], [], [], 0.0
        
        # Step 1.1: Find candidate edges for start and destination
        # Select up to 3 closest candidate edges (by distance)
        # If fewer than 3 candidates found: Use all available candidates
        start_lon, start_lat = segment[0]
        dest_lon, dest_lat = segment[-1]
        
        # Find all candidate edges within 100m radius for start and destination
        start_candidates = self._find_candidate_edges(start_lon, start_lat, max_radius=100.0)
        dest_candidates = self._find_candidate_edges(dest_lon, dest_lat, max_radius=100.0)
        
        # Always return candidates even if route calculation fails
        if not start_candidates:
            self.log(f"⚠️ Segment {seg_idx + 1}: No candidate edges found for start GPS point")
            return [], dest_candidates if dest_candidates else [], [], [], 0.0
        
        if not dest_candidates:
            self.log(f"⚠️ Segment {seg_idx + 1}: No candidate edges found for destination GPS point")
            return [], start_candidates, [], [], 0.0
        
        self.log(f"Segment {seg_idx + 1} Step 1.1: Found {len(start_candidates)} start candidates, {len(dest_candidates)} dest candidates")
        
        # Step 1.2: Calculate k-shortest paths for all combinations (up to 5 routes per combination)
        all_routes = []  # List of (route_edges, cost) - will contain up to 5 routes per combination
        
        for start_edge_id, start_dist in start_candidates:
            for dest_edge_id, dest_dist in dest_candidates:
                try:
                    # Get base edge IDs (remove lane suffix)
                    start_base_id = start_edge_id.split('#')[0]
                    dest_base_id = dest_edge_id.split('#')[0]
                    
                    # Check if edges exist in the network before trying to get them
                    if not sumo_net.hasEdge(start_base_id):
                        continue  # Skip if edge doesn't exist
                    if not sumo_net.hasEdge(dest_base_id):
                        continue  # Skip if edge doesn't exist
                    
                    # Get SUMO edge objects
                    start_edge = sumo_net.getEdge(start_base_id)
                    dest_edge = sumo_net.getEdge(dest_base_id)
                    
                    # Calculate k-shortest paths (up to 5 routes)
                    # Check if getKShortestPaths method exists
                    if hasattr(sumo_net, 'getKShortestPaths'):
                        try:
                            k_routes = sumo_net.getKShortestPaths(start_edge, dest_edge, 5)
                            if k_routes:
                                # getKShortestPaths returns a list of (edges, cost) tuples
                                for route_result in k_routes:
                                    if route_result and len(route_result) >= 2:
                                        route_edges, cost = route_result
                                        if route_edges:
                                            # Convert to list of edge IDs (base IDs only)
                                            edge_ids = [edge.getID() for edge in route_edges]
                                            all_routes.append((edge_ids, cost))
                        except Exception as e:
                            # Fallback to single shortest path
                            route_result = sumo_net.getShortestPath(start_edge, dest_edge)
                            if route_result and len(route_result) >= 2:
                                route_edges, cost = route_result
                                if route_edges:
                                    edge_ids = [edge.getID() for edge in route_edges]
                                    all_routes.append((edge_ids, cost))
                    else:
                        # Fallback: use getShortestPath (only one route)
                        route_result = sumo_net.getShortestPath(start_edge, dest_edge)
                        if route_result and len(route_result) >= 2:
                            route_edges, cost = route_result
                            if route_edges:
                                edge_ids = [edge.getID() for edge in route_edges]
                                all_routes.append((edge_ids, cost))
                except Exception as e:
                    # Skip this combination if route calculation fails (don't log every error to avoid spam)
                    continue
        
        if not all_routes:
            self.log(f"⚠️ Segment {seg_idx + 1}: No valid routes found for any combination")
            # Return empty route but still return candidates for visualization
            return [], start_candidates, dest_candidates, [], 0.0
        
        # Step 1.3: Select top routes (up to 5, sorted by cost) and calculate similarity scores
        all_routes.sort(key=lambda x: x[1])  # Sort by cost
        # Deduplicate routes (same edge sequence) and take top 5 unique routes
        seen_routes = set()
        top_routes = []
        for route_edges, cost in all_routes:
            route_tuple = tuple(route_edges)  # Use tuple for hashing
            if route_tuple not in seen_routes:
                seen_routes.add(route_tuple)
                top_routes.append((route_edges, cost))
                if len(top_routes) >= 5:
                    break
        
        # Step 1.3: Calculate similarity scores for all top routes and select the best one
        # Important: Use the current segment (may be shortened) for similarity calculation
        # This ensures only GPS points actually used for route calculation are scored
        route_scores = []
        for route_edges, cost in top_routes:
            # Calculate similarity using the segment that was used to find this route
            # (segment may be shortened if iterative shortening occurred)
            similarity_score = self._calculate_route_similarity_score(route_edges, segment, seg_idx)
            route_scores.append((route_edges, cost, similarity_score))
        
        # Select route with highest similarity score
        if route_scores:
            # Sort by similarity score (descending), then by cost (ascending) as tiebreaker
            route_scores.sort(key=lambda x: (-x[2], x[1]))
            best_route, best_cost, best_score = route_scores[0]
            
            self.log(f"Segment {seg_idx + 1} Step 1.3: Evaluated {len(route_scores)} route(s), selected route with similarity score {best_score:.4f} (cost: {best_cost:.2f})")
            
            # Log all route scores for debugging
            for idx, (route_edges, cost, score) in enumerate(route_scores):
                self.log(f"  Route {idx + 1}: similarity={score:.4f}, cost={cost:.2f}, edges={len(route_edges)}")
            
            # Return routes sorted by similarity score (best first) for visualization
            # The first route is the selected one (highest similarity score)
            selected_routes = [route_edges for route_edges, _, _ in route_scores]
        else:
            selected_routes = []
            best_score = 0.0
        
        # Step 1.4: Store destination candidate edges
        dest_candidate_edge_ids = [edge_id.split('#')[0] for edge_id, _ in dest_candidates]
        self.destination_candidate_edges = dest_candidate_edge_ids[:3]  # Store up to 3
        
        # Step 1.5: First edge is already marked as valid (it's the first edge of selected route)
        
        # Return routes sorted by similarity score (best first) along with their scores
        # The first route in the list has the highest similarity score and is the selected route
        route_scores_list = [score for _, _, score in route_scores] if route_scores else []
        return selected_routes, start_candidates, dest_candidates, route_scores_list, best_score
    
    def _calculate_step1_route(self, segment: List[List[float]], sumo_net, seg_idx: int) -> Tuple[List[str], List[Tuple[str, float]], List[Tuple[str, float]], List[float], List[List[float]]]:
        """
        Step 1: Calculate initial SUMO route from start to destination with iterative shortening.
        
        If no route found or similarity score < 0.75, iteratively shortens the segment by removing
        GPS points: 6 points from start, then 4 points from end, repeating (max 10 iterations or until < 4 points remain).
        
        Args:
            segment: List of [lon, lat] GPS points
            sumo_net: SUMO network object
            seg_idx: Segment index for logging
        
        Returns:
            Tuple of (route_edges, start_candidates, dest_candidates, route_scores, final_segment):
            - route_edges: List of edge IDs representing the route, or empty list if no route found
            - start_candidates: List of (edge_id, distance) tuples for start point
            - dest_candidates: List of (edge_id, distance) tuples for destination point
            - route_scores: List of similarity scores for each route (sorted by score, best first)
            - final_segment: The final segment used (may be shortened from original)
        """
        if len(segment) < 2:
            return [], [], [], [], segment
        
        SIMILARITY_THRESHOLD = 0.75
        MAX_ITERATIONS = 10
        MIN_POINTS = 4
        
        current_segment = segment.copy()
        removed_from_start = 0
        removed_from_end = 0
        best_result = None
        best_score = 0.0
        best_segment = current_segment
        
        # Try initial calculation
        routes, start_candidates, dest_candidates, route_scores, score = self._calculate_step1_route_single(
            current_segment, sumo_net, seg_idx
        )
        
        if routes and score >= SIMILARITY_THRESHOLD:
            # Success on first try
            self.log(f"✓ Segment {seg_idx + 1} Step 1: Route found with similarity score {score:.4f} (≥ {SIMILARITY_THRESHOLD})")
            # Return a copy to ensure the segment matches the route
            return routes, start_candidates, dest_candidates, route_scores, current_segment.copy()
        
        # Store initial result as best so far
        if routes:
            # Store the segment that was used to calculate this route
            best_result = (routes, start_candidates, dest_candidates, route_scores, current_segment.copy())
            best_score = score
            best_segment = current_segment.copy()
            reason = "low similarity score" if score < SIMILARITY_THRESHOLD else "no route found"
            self.log(f"⚠️ Segment {seg_idx + 1} Step 1: Initial attempt - {reason} (score: {score:.4f}), starting iterative shortening...")
        else:
            self.log(f"⚠️ Segment {seg_idx + 1} Step 1: No route found, starting iterative shortening...")
        
        # Iterative shortening: Remove 6 points from start, then 4 from end, repeat
        POINTS_TO_REMOVE_FROM_START = 6
        POINTS_TO_REMOVE_FROM_END = 4
        
        iteration = 0
        while iteration < MAX_ITERATIONS:
            if len(current_segment) < MIN_POINTS:
                self.log(f"⚠️ Segment {seg_idx + 1} Step 1: Stopping - too few GPS points remaining ({len(current_segment)} < {MIN_POINTS})")
                break
            
            # Remove 6 points from start
            points_removed_this_iteration = 0
            for i in range(POINTS_TO_REMOVE_FROM_START):
                if len(current_segment) < MIN_POINTS:
                    break
                if len(current_segment) <= 1:
                    break
                removed_point = current_segment.pop(0)
                removed_from_start += 1
                points_removed_this_iteration += 1
                self.log(f"  Iteration {iteration + 1}: Removed GPS point from START (point {removed_from_start}): [{removed_point[0]:.6f}, {removed_point[1]:.6f}]")
            
            if len(current_segment) < MIN_POINTS:
                break
            
            # Recalculate after removing from start
            routes, start_candidates, dest_candidates, route_scores, score = self._calculate_step1_route_single(
                current_segment, sumo_net, seg_idx
            )
            
            if routes:
                if score > best_score:
                    # Save the best result along with the segment that was used to calculate it
                    best_result = (routes, start_candidates, dest_candidates, route_scores, current_segment.copy())
                    best_score = score
                    best_segment = current_segment.copy()
                
                if score >= SIMILARITY_THRESHOLD:
                    self.log(f"✓ Segment {seg_idx + 1} Step 1: Success after removing {removed_from_start} point(s) from start! Similarity score: {score:.4f} (≥ {SIMILARITY_THRESHOLD})")
                    return routes, start_candidates, dest_candidates, route_scores, current_segment.copy()
                else:
                    self.log(f"  Iteration {iteration + 1} (after removing {points_removed_this_iteration} from start): Route found but score {score:.4f} < {SIMILARITY_THRESHOLD}, continuing...")
            else:
                self.log(f"  Iteration {iteration + 1} (after removing {points_removed_this_iteration} from start): No route found, continuing...")
            
            if len(current_segment) < MIN_POINTS:
                break
            
            # Remove 4 points from end
            points_removed_this_iteration = 0
            for i in range(POINTS_TO_REMOVE_FROM_END):
                if len(current_segment) < MIN_POINTS:
                    break
                if len(current_segment) <= 1:
                    break
                removed_point = current_segment.pop()
                removed_from_end += 1
                points_removed_this_iteration += 1
                self.log(f"  Iteration {iteration + 1}: Removed GPS point from END (point {removed_from_end}): [{removed_point[0]:.6f}, {removed_point[1]:.6f}]")
            
            if len(current_segment) < MIN_POINTS:
                break
            
            # Recalculate after removing from end
            routes, start_candidates, dest_candidates, route_scores, score = self._calculate_step1_route_single(
                current_segment, sumo_net, seg_idx
            )
            
            if routes:
                if score > best_score:
                    # Save the best result along with the segment that was used to calculate it
                    best_result = (routes, start_candidates, dest_candidates, route_scores, current_segment.copy())
                    best_score = score
                    best_segment = current_segment.copy()
                
                if score >= SIMILARITY_THRESHOLD:
                    self.log(f"✓ Segment {seg_idx + 1} Step 1: Success after removing {removed_from_start} point(s) from start and {removed_from_end} point(s) from end! Similarity score: {score:.4f} (≥ {SIMILARITY_THRESHOLD})")
                    return routes, start_candidates, dest_candidates, route_scores, current_segment.copy()
                else:
                    self.log(f"  Iteration {iteration + 1} (after removing {points_removed_this_iteration} from end): Route found but score {score:.4f} < {SIMILARITY_THRESHOLD}, continuing...")
            else:
                self.log(f"  Iteration {iteration + 1} (after removing {points_removed_this_iteration} from end): No route found, continuing...")
            
            iteration += 1
            
            # Recalculate Step 1 with shortened segment
            routes, start_candidates, dest_candidates, route_scores, score = self._calculate_step1_route_single(
                current_segment, sumo_net, seg_idx
            )
            
            if routes:
                if score > best_score:
                    # Save the best result along with the segment that was used to calculate it
                    best_result = (routes, start_candidates, dest_candidates, route_scores, current_segment.copy())
                    best_score = score
                    best_segment = current_segment.copy()
                
                if score >= SIMILARITY_THRESHOLD:
                    self.log(f"✓ Segment {seg_idx + 1} Step 1: Success after {iteration + 1} iteration(s)! Similarity score: {score:.4f} (≥ {SIMILARITY_THRESHOLD})")
                    self.log(f"  Removed {removed_from_start} point(s) from start, {removed_from_end} point(s) from end")
                    # Return the segment that was used to calculate this route
                    return routes, start_candidates, dest_candidates, route_scores, current_segment.copy()
                else:
                    self.log(f"  Iteration {iteration + 1}: Route found but score {score:.4f} < {SIMILARITY_THRESHOLD}, continuing...")
            else:
                self.log(f"  Iteration {iteration + 1}: No route found, continuing...")
        
        # If we get here, we didn't reach the threshold
        if best_result:
            routes, start_candidates, dest_candidates, route_scores, best_segment_for_route = best_result
            self.log(f"⚠️ Segment {seg_idx + 1} Step 1: Best route found after {MAX_ITERATIONS} iterations with score {best_score:.4f} (< {SIMILARITY_THRESHOLD})")
            self.log(f"  Removed {removed_from_start} point(s) from start, {removed_from_end} point(s) from end")
            self.log(f"  Step 1 failed - will proceed to Step 2")
            # Return the segment that was used to calculate the best route
            return routes, start_candidates, dest_candidates, route_scores, best_segment_for_route
        else:
            self.log(f"⚠️ Segment {seg_idx + 1} Step 1: No route found after {MAX_ITERATIONS} iterations")
            self.log(f"  Removed {removed_from_start} point(s) from start, {removed_from_end} point(s) from end")
            self.log(f"  Step 1 failed - will proceed to Step 2")
            return [], start_candidates, dest_candidates, [], current_segment
    
    def _draw_gps_points_as_stars(self, segments: List[List[List[float]]]):
        """
        Draw stars at each GPS point converted to SUMO coordinates.
        This helps verify alignment between GPS points and SUMO network.
        
        Args:
            segments: List of polyline segments (each segment is a list of [lon, lat] points)
        """
        if not self.network_parser:
            self.log("⚠️ Network parser not available for GPS point drawing")
            return
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
        from PySide6.QtWidgets import (QGraphicsPathItem, QGraphicsRectItem,
                                       QGraphicsTextItem)

        # Star color - bright yellow for visibility
        star_color = QColor(255, 255, 0)  # Yellow
        star_pen = QPen(star_color, 2.0)  # Thicker pen for visibility
        star_brush = QBrush(star_color)
        
        # Star size (radius in pixels) - increased for visibility
        star_radius = 8.0
        
        # Text settings for numbers - readable font size
        text_font = QFont("Arial", 10, QFont.Bold)
        text_color = QColor(0, 0, 0)  # Black text for contrast
        
        def create_star_path(center_x: float, center_y: float, radius: float) -> QPainterPath:
            """Create a star shape path."""
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
        
        total_points = 0
        point_counter = 0  # Global counter across all segments (starts from 0)
        ox = self.map_offset_x_spinbox.value()
        oy = self.map_offset_y_spinbox.value()
        
        for seg_idx, segment in enumerate(segments):
            if not segment:
                continue
            
            for point_idx, (lon, lat) in enumerate(segment):
                # Apply GPS offset for display (trajectory moves, map/network stay fixed)
                adj_lon, adj_lat = apply_gps_offset(lon, lat, ox, oy)
                # Convert GPS to SUMO coordinates
                sumo_coords = self.network_parser.gps_to_sumo_coords(adj_lon, adj_lat)
                if not sumo_coords:
                    continue
                
                x, y = sumo_coords
                
                # Flip Y coordinate to match network display
                y_flipped = flip_y(y)
                
                # Create star path
                star_path = create_star_path(x, y_flipped, star_radius)
                
                # Create graphics item for star
                star_item = QGraphicsPathItem(star_path)
                star_item.setPen(star_pen)
                star_item.setBrush(star_brush)
                star_item.setZValue(2000)  # Draw on top of everything
                star_item.setVisible(True)  # Ensure visibility
                try:
                    self.map_view.scene.addItem(star_item)
                    self._route_items.append(star_item)
                except RuntimeError:
                    # Scene was deleted, skip
                    continue
                
                # Create text item for number (simplified - no background for now)
                text_str = str(point_counter)
                text_item = QGraphicsTextItem(text_str)
                text_item.setFont(text_font)
                text_item.setDefaultTextColor(text_color)
                
                # Get text bounding rect (need to set font first)
                text_rect = text_item.boundingRect()
                
                # Center text on star
                text_x = x - text_rect.width() / 2
                text_y = y_flipped - text_rect.height() / 2
                text_item.setPos(text_x, text_y)
                text_item.setZValue(2001)  # Above star
                
                # Add text item
                try:
                    self.map_view.scene.addItem(text_item)
                    self._route_items.append(text_item)
                except RuntimeError:
                    # Scene was deleted, skip
                    continue
                
                point_counter += 1
                total_points += 1
        
        self.log(f"✅ Drew {total_points} GPS points as numbered stars (using SUMO coordinates)")
    
    def _draw_route_edges(self, route_edges: List[str], color: QColor = None):
        """
        Draw route edges on the map as dashed lines.
        
        Args:
            route_edges: List of edge IDs to draw
            color: QColor to use for drawing (default: magenta)
        """
        if not self.network_parser:
            return
        
        edges_dict = self.network_parser.get_edges()
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPen
        from PySide6.QtWidgets import QGraphicsLineItem

        # Use provided color or default to magenta
        if color is None:
            route_color = QColor(255, 0, 255)  # Magenta
        else:
            route_color = color
        
        route_pen = QPen(route_color)
        route_pen.setWidth(3)
        route_pen.setStyle(Qt.DashLine)
        route_pen.setDashPattern([5, 5])  # 5px dash, 5px gap
        
        for edge_id in route_edges:
            # Try base edge ID first, then with lane suffixes
            edge_data = None
            if edge_id in edges_dict:
                edge_data = edges_dict[edge_id]
            else:
                # Try with lane suffix #0
                if f"{edge_id}#0" in edges_dict:
                    edge_data = edges_dict[f"{edge_id}#0"]
            
            if not edge_data:
                continue
            
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            # Use first lane's shape
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Draw line segments along the edge shape
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                
                # Flip Y coordinates
                y1_flipped = flip_y(y1)
                y2_flipped = flip_y(y2)
                
                line = QGraphicsLineItem(x1, y1_flipped, x2, y2_flipped)
                line.setPen(route_pen)
                line.setZValue(1000)  # Draw on top
                self.map_view.scene.addItem(line)
                self._route_items.append(line)
    
    def _find_candidate_edges_step2(self, lon: float, lat: float, max_candidates: int = 10, max_radius: float = 300.0) -> List[Tuple[str, float]]:
        """
        Step 2: Find candidate edges for a GPS point.
        
        Uses the SAME logic as Step 1's _find_candidate_edges to ensure consistency.
        The only difference is that we don't use max_radius filtering (we want to see all closest edges).
        
        Args:
            lon: Longitude
            lat: Latitude
            max_candidates: Maximum number of candidates to return (default: 10)
            max_radius: Not used (for compatibility only)
        
        Returns:
            List of (edge_id, distance) tuples, sorted by distance (closest first)
        """
        if not self.network_parser:
            return []
        
        # Use the SAME method as Step 1, but with a larger search radius to ensure we find all edges
        # Step 1 uses spatial index with 500m radius, so we'll use the same approach
        # but iterate through all edges to ensure we don't miss anything
        
        # Convert GPS to SUMO coordinates
        sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
        if not sumo_coords:
            return []
        
        x, y = sumo_coords
        
        # Check ALL edges from the network (not just spatial index candidates)
        # This ensures we find edges that might be outside the spatial index radius
        # The spatial index uses 500m radius, but we want to find ALL closest edges
        edges_dict = self.network_parser.get_edges()
        edge_distances = []  # List of (edge_id, distance)
        
        import math

        # Iterate through ALL edges to ensure we don't miss anything
        # This is different from Step 1 which uses spatial index for performance
        # Step 2 needs to be thorough and find ALL edges, not just those in a radius
        edges_to_check = edges_dict.keys()
        
        for edge_id in edges_to_check:
            if edge_id not in edges_dict:
                continue
            
            edge_data = edges_dict[edge_id]
            lanes = edge_data.get('lanes', [])
            
            # Use SAME logic as Step 1: skip edges without lanes
            if not lanes:
                continue
            
            # Use first lane's shape (same as Step 1)
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Calculate minimum distance from point to edge shape (SAME as Step 1)
            min_dist = float('inf')
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                # Calculate distance manually (point to segment)
                dx = x2 - x1
                dy = y2 - y1
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                else:
                    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                    closest_x = x1 + t * dx
                    closest_y = y1 + t * dy
                    dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                min_dist = min(min_dist, dist)
            
            # Include all edges (same as Step 1)
            if min_dist < float('inf'):
                edge_distances.append((edge_id, min_dist))
        
        # Deduplicate by base edge ID (SAME as Step 1)
        base_edge_map = {}
        for edge_id, distance in edge_distances:
            # Get base edge ID (strip # suffix) - SAME as Step 1
            base_id = edge_id.split('#')[0] if '#' in edge_id else edge_id
            
            # Keep only closest variant per base edge (SAME as Step 1)
            if base_id not in base_edge_map or distance < base_edge_map[base_id][1]:
                base_edge_map[base_id] = (edge_id, distance)
        
        # Sort by distance (SAME as Step 1)
        deduplicated = sorted(base_edge_map.values(), key=lambda x: x[1])
        
        # Debug: Log for first GPS point
        if not hasattr(self, '_step2_coord_logged'):
            self.log(f"📍 Step 2: First GPS point ({lon:.6f}, {lat:.6f}) → SUMO ({x:.2f}, {y:.2f})")
            self.log(f"📍 Step 2: Checking ALL {len(edges_to_check)} edges from network (no spatial index filtering)")
            
            # DIAGNOSTIC: Compare with Step 1's spatial index approach
            spatial_index_candidates = self._edge_spatial_index.get_candidates_in_radius(x, y, radius=500.0)
            self.log(f"🔍 DIAGNOSTIC: Step 1 spatial index (500m) would find {len(spatial_index_candidates)} candidate edges")
            if hasattr(self, '_step1_route_edges') and self._step1_route_edges:
                # Check how many Step 1 route edges are in spatial index
                step1_in_spatial_index = 0
                for step1_base_id in self._step1_route_edges:
                    for eid in spatial_index_candidates:
                        base_id = eid.split('#')[0] if '#' in eid else eid
                        if base_id == step1_base_id:
                            step1_in_spatial_index += 1
                            break
                self.log(f"🔍 DIAGNOSTIC: {step1_in_spatial_index}/{len(self._step1_route_edges)} Step 1 route edges are in spatial index (500m radius)")
                if step1_in_spatial_index < len(self._step1_route_edges):
                    self.log(f"⚠️ DIAGNOSTIC: {len(self._step1_route_edges) - step1_in_spatial_index} Step 1 route edges are NOT in spatial index - they may be beyond 500m!")
            self._step2_coord_logged = True
        
        if deduplicated:
            top_50 = deduplicated[:50]
            top_50_base_ids = {eid.split('#')[0] if '#' in eid else eid for eid, _ in top_50}
            
            # Check if Step 1 route edges are in Step 2 results
            step1_edges_to_include = []
            if hasattr(self, '_step1_route_edges') and self._step1_route_edges:
                # Debug: log that we're checking
                if not hasattr(self, '_step2_comparison_logged'):
                    self.log(f"🔍 Step 2: Comparing against {len(self._step1_route_edges)} Step 1 route edges: {sorted(list(self._step1_route_edges))[:10]}{'...' if len(self._step1_route_edges) > 10 else ''}")
                    self._step2_comparison_logged = True
                # Find which Step 1 route edges are in the full deduplicated list
                all_base_ids_map = {}  # base_id -> (rank, distance, edge_id)
                for idx, (eid, dist) in enumerate(deduplicated):
                    base_id = eid.split('#')[0] if '#' in eid else eid
                    if base_id not in all_base_ids_map:
                        all_base_ids_map[base_id] = (idx + 1, dist, eid)
                
                # Check each Step 1 route edge
                found_step1_edges = []
                missing_step1_edges = []
                for step1_base_id in self._step1_route_edges:
                    if step1_base_id in all_base_ids_map:
                        rank, dist, eid = all_base_ids_map[step1_base_id]
                        found_step1_edges.append((step1_base_id, rank, dist))
                        # Include ALL Step 1 route edges regardless of distance
                        # They're part of the route and should be visible even if far from GPS points
                        step1_edges_to_include.append((eid, dist))
                    else:
                        missing_step1_edges.append(step1_base_id)
                        # DIAGNOSTIC: Why is this Step 1 route edge missing?
                        # Check if edge exists in network but wasn't processed
                        edge_found_in_network = False
                        edge_has_lanes = False
                        edge_has_shape = False
                        for check_eid, check_edata in edges_dict.items():
                            check_base_id = check_eid.split('#')[0] if '#' in check_eid else check_eid
                            if check_base_id == step1_base_id:
                                edge_found_in_network = True
                                lanes = check_edata.get('lanes', [])
                                if lanes:
                                    edge_has_lanes = True
                                    shape = lanes[0].get('shape', [])
                                    if len(shape) >= 2:
                                        edge_has_shape = True
                                        # Calculate distance for this missing edge
                                        min_dist = float('inf')
                                        for i in range(len(shape) - 1):
                                            x1, y1 = shape[i]
                                            x2, y2 = shape[i + 1]
                                            dx = x2 - x1
                                            dy = y2 - y1
                                            len_sq = dx * dx + dy * dy
                                            if len_sq == 0:
                                                dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                                            else:
                                                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                                                closest_x = x1 + t * dx
                                                closest_y = y1 + t * dy
                                                dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                                            min_dist = min(min_dist, dist)
                                        # Log diagnostic info
                                        if not hasattr(self, '_step2_missing_diagnostic_logged'):
                                            self.log(f"🔍 DIAGNOSTIC: Missing Step 1 edge '{step1_base_id}': Found in network={edge_found_in_network}, Has lanes={edge_has_lanes}, Has shape={edge_has_shape}, Calculated distance={min_dist:.2f}m, GPS point SUMO coords=({x:.2f}, {y:.2f}), Edge shape first point=({shape[0][0]:.2f}, {shape[0][1]:.2f})")
                                            self._step2_missing_diagnostic_logged = True
                                break
                        if not edge_found_in_network and not hasattr(self, '_step2_missing_diagnostic_logged'):
                            self.log(f"🔍 DIAGNOSTIC: Missing Step 1 edge '{step1_base_id}': NOT FOUND IN NETWORK AT ALL")
                            self._step2_missing_diagnostic_logged = True
                
                # Log findings with detailed diagnostics
                if found_step1_edges:
                    # Sort by rank
                    found_step1_edges.sort(key=lambda x: x[1])
                    top_found = found_step1_edges[:5]
                    self.log(f"✅ Step 2 GPS ({lon:.6f}, {lat:.6f}): Found {len(found_step1_edges)}/{len(self._step1_route_edges)} Step 1 route edges: {[(bid, f'rank {r}, {d:.2f}m') for bid, r, d in top_found]}{'...' if len(found_step1_edges) > 5 else ''}")
                    
                    # DIAGNOSTIC: For the first GPS point, check why Step 1 route edges are ranked so low
                    if not hasattr(self, '_step2_distance_diagnostic_logged'):
                        self.log(f"🔍 DIAGNOSTIC: Analyzing why Step 1 route edges are ranked low...")
                        self.log(f"   GPS point SUMO coordinates: ({x:.2f}, {y:.2f})")
                        self.log(f"   Total edges checked: {len(edges_to_check)}")
                        self.log(f"   Total edges with valid shape data: {len(deduplicated)}")
                        
                        # Check a few Step 1 route edges that are ranked low
                        for bid, rank, dist in found_step1_edges[:3]:
                            # Find the actual edge in the network
                            for check_eid, check_edata in edges_dict.items():
                                check_base_id = check_eid.split('#')[0] if '#' in check_eid else check_eid
                                if check_base_id == bid:
                                    lanes = check_edata.get('lanes', [])
                                    if lanes:
                                        shape = lanes[0].get('shape', [])
                                        if len(shape) >= 2:
                                            self.log(f"   Step 1 edge '{bid}' (rank {rank}, {dist:.2f}m):")
                                            self.log(f"      Edge ID: {check_eid}")
                                            self.log(f"      Shape points: {len(shape)}")
                                            self.log(f"      First shape point: ({shape[0][0]:.2f}, {shape[0][1]:.2f})")
                                            self.log(f"      Last shape point: ({shape[-1][0]:.2f}, {shape[-1][1]:.2f})")
                                            # Check if this edge would be found by Step 1's spatial index
                                            spatial_index_candidates = self._edge_spatial_index.get_candidates_in_radius(x, y, radius=500.0)
                                            found_by_spatial_index = check_eid in spatial_index_candidates
                                            self.log(f"      Found by Step 1 spatial index (500m): {found_by_spatial_index}")
                                            if found_by_spatial_index:
                                                self.log(f"      ⚠️ Edge IS in spatial index but ranked {rank} in Step 2 - distance calculation may be wrong!")
                                    break
                        self._step2_distance_diagnostic_logged = True
                
                if missing_step1_edges:
                    self.log(f"⚠️ Step 2 GPS ({lon:.6f}, {lat:.6f}): Missing {len(missing_step1_edges)} Step 1 route edges: {missing_step1_edges[:10]}{'...' if len(missing_step1_edges) > 10 else ''}")
            
            self.log(f"🔍 Step 2 candidate search for GPS ({lon:.6f}, {lat:.6f}): Found {len(deduplicated)} unique base edges, top 50: {[(eid.split('#')[0] if '#' in eid else eid, f'{d:.2f}m') for eid, d in top_50]}")
        
        # Apply radius filter: include edges within max_radius
        filtered_by_radius = [(eid, dist) for eid, dist in deduplicated if dist <= max_radius]
        
        # Also include ALL Step 1 route edges (even if beyond radius)
        # They're part of the route and should be visible even if far from GPS points
        # Deduplicate: don't add Step 1 edges that are already in filtered_by_radius
        filtered_base_ids = {eid.split('#')[0] if '#' in eid else eid for eid, _ in filtered_by_radius}
        step1_edges_added = []
        for eid, dist in step1_edges_to_include:
            base_id = eid.split('#')[0] if '#' in eid else eid
            if base_id not in filtered_base_ids:
                step1_edges_added.append((eid, dist))
                filtered_base_ids.add(base_id)
        
        # Sort regular edges by distance
        filtered_by_radius.sort(key=lambda x: x[1])
        
        # Return regular edges (up to max_candidates) PLUS all Step 1 route edges
        # This ensures Step 1 route edges are always visible even if beyond radius
        result = filtered_by_radius[:max_candidates]
        regular_count = len(result)
        
        # Add Step 1 route edges (they may push total above max_candidates, but that's OK)
        # Sort Step 1 edges by distance before adding
        step1_edges_added.sort(key=lambda x: x[1])
        result.extend(step1_edges_added)
        step1_count = len(step1_edges_added)
        
        # Final sort to maintain distance order
        result.sort(key=lambda x: x[1])
        
        # Log summary for first GPS point
        if not hasattr(self, '_step2_return_logged'):
            self.log(f"📊 Step 2 candidate return: {regular_count} regular edges (within {max_radius}m) + {step1_count} Step 1 route edges = {len(result)} total candidates")
            self._step2_return_logged = True
        
        return result
    
    def _draw_step2_gps_points_and_edges(self, segments: List[List[List[float]]]):
        """
        Step 2: Draw each GPS point and its closest edges (up to 10 within 300m) with the same distinct color.
        
        For each GPS point in the trimmed segments:
        - Find up to 10 closest candidate edges within 300m radius (without spatial index)
        - Assign a distinct color to the GPS point and its edges
        - Draw the GPS point and edges with that color
        - If an edge is selected by multiple GPS points, draw it with a dashed line
        
        Args:
            segments: List of polyline segments (each segment is a list of [lon, lat] points)
        """
        if not self.network_parser:
            self.log("⚠️ Network parser not available for Step 2")
            return
        
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor, QPen
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem

        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        edges_dict = self.network_parser.get_edges()
        total_points = 0
        
        # Track which edges are selected by which GPS points
        # base_edge_id -> list of (point_index, color)
        edge_to_points = {}
        point_colors = []
        point_candidates = []  # List of (point_index, candidates_list)
        
        # Generate distinct colors using HSV color space for better color distribution
        import colorsys

        # First pass: collect all GPS points and their candidate edges
        for seg_idx, segment in enumerate(segments):
            if not segment:
                continue
            
            for point_idx, (lon, lat) in enumerate(segment):
                # Generate a distinct color for this GPS point
                # Use HSV to get evenly distributed colors
                hue = (total_points * 137.508) % 360  # Golden angle approximation for good distribution
                saturation = 0.7
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
                point_color = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                point_colors.append(point_color)
                
                # Find candidate edges for this GPS point (includes Step 1 route edges even if beyond 300m)
                candidates = self._find_candidate_edges_step2(lon, lat, max_candidates=10, max_radius=300.0)
                
                if not candidates:
                    point_colors.pop()  # Remove the color we just added
                    continue
                
                point_candidates.append((total_points, candidates))
                
                # Log for first few points to verify Step 1 edges are included
                if total_points < 3:
                    step1_in_candidates = sum(1 for eid, _ in candidates if eid.split('#')[0] in (self._step1_route_edges if hasattr(self, '_step1_route_edges') else set()))
                    self.log(f"📍 Step 2 GPS point {total_points}: {len(candidates)} candidates returned ({step1_in_candidates} are Step 1 route edges)")
                
                # Track which edges are selected by this point (use base ID)
                for edge_id, distance in candidates:
                    base_edge_id = edge_id.split('#')[0]  # Use base ID to track duplicates
                    if base_edge_id not in edge_to_points:
                        edge_to_points[base_edge_id] = []
                    edge_to_points[base_edge_id].append((total_points, point_color))
                
                total_points += 1
        
        # Track which edges have been drawn (to avoid duplicates for multi-color edges)
        drawn_edges = set()
        
        # Second pass: draw GPS points and edges
        point_index = 0
        for seg_idx, segment in enumerate(segments):
            if not segment:
                continue
            
            for point_idx, (lon, lat) in enumerate(segment):
                if point_index >= len(point_colors):
                    break
                
                point_color = point_colors[point_index]
                
                # Find candidates for this point
                candidates = None
                for pc_idx, pc_candidates in point_candidates:
                    if pc_idx == point_index:
                        candidates = pc_candidates
                        break
                
                if not candidates:
                    point_index += 1
                    continue
                
                # Convert GPS to SUMO coordinates for the point
                sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
                if not sumo_coords:
                    point_index += 1
                    continue
                
                x, y = sumo_coords
                y_flipped = flip_y(y)
                
                # Draw GPS point as a colored circle
                point_radius = 8.0
                point_item = QGraphicsEllipseItem(x - point_radius, y_flipped - point_radius, 
                                                  point_radius * 2, point_radius * 2)
                point_pen = QPen(point_color)
                point_pen.setWidth(2)
                point_brush = QBrush(point_color)
                point_item.setPen(point_pen)
                point_item.setBrush(point_brush)
                point_item.setZValue(2000)  # Draw on top
                self.map_view.scene.addItem(point_item)
                self._route_items.append(point_item)
                
                # Draw edges for this point
                for edge_id, distance in candidates:
                    base_edge_id = edge_id.split('#')[0]
                    
                    # Check if this edge is selected by multiple points
                    selecting_points = edge_to_points.get(base_edge_id, [])
                    is_multiple = len(selecting_points) > 1
                    
                    # For edges selected by multiple points, only draw once with all colors
                    if is_multiple and base_edge_id in drawn_edges:
                        continue  # Skip - will be drawn with multi-color pattern
                    
                    # Try to find edge in edges_dict
                    edge_data = None
                    if edge_id in edges_dict:
                        edge_data = edges_dict[edge_id]
                    else:
                        # Try with lane suffix #0
                        if f"{edge_id}#0" in edges_dict:
                            edge_data = edges_dict[f"{edge_id}#0"]
                        else:
                            # Try base edge ID (without any suffix)
                            base_edge_id = edge_id.split('#')[0]
                            # Search for any variant of this base edge
                            for eid, edata in edges_dict.items():
                                if eid.split('#')[0] == base_edge_id:
                                    edge_data = edata
                                    break
                    
                    if not edge_data:
                        # Log missing edge for debugging
                        self.log(f"⚠️ Step 2: Edge {edge_id} (base: {base_edge_id}) not found in edges_dict")
                        continue
                    
                    lanes = edge_data.get('lanes', [])
                    if not lanes:
                        continue
                    
                    # Use first lane's shape
                    shape = lanes[0].get('shape', [])
                    if len(shape) < 2:
                        continue
                    
                    if is_multiple:
                        # Draw edge with multi-color striped pattern
                        # Get all colors from selecting points
                        selecting_colors = [color for _, color in selecting_points]
                        num_colors = len(selecting_colors)
                        
                        # Draw the edge as segments, alternating colors to create a striped pattern
                        # This creates a clear blue-red-yellow (or however many colors) striped effect
                        import math
                        segment_length_pixels = 12  # Length of each color segment in pixels
                        
                        # Draw each shape segment, dividing it into color segments
                        for seg_idx in range(len(shape) - 1):
                            x1, y1 = shape[seg_idx]
                            x2, y2 = shape[seg_idx + 1]
                            
                            # Flip Y coordinates
                            y1_flipped = flip_y(y1)
                            y2_flipped = flip_y(y2)
                            
                            # Calculate segment vector
                            dx = x2 - x1
                            dy = y2_flipped - y1_flipped
                            segment_length = math.sqrt(dx * dx + dy * dy)
                            
                            if segment_length < 0.1:
                                continue
                            
                            # Normalize direction vector
                            unit_x = dx / segment_length
                            unit_y = dy / segment_length
                            
                            # Draw segments with alternating colors
                            current_pos = 0.0
                            color_idx = 0
                            
                            while current_pos < segment_length:
                                # Calculate segment end position
                                segment_end = min(current_pos + segment_length_pixels, segment_length)
                                
                                # Get color for this segment (alternate through all colors)
                                color = selecting_colors[color_idx % num_colors]
                                
                                # Calculate segment start and end points
                                seg_x1 = x1 + current_pos * unit_x
                                seg_y1 = y1_flipped + current_pos * unit_y
                                seg_x2 = x1 + segment_end * unit_x
                                seg_y2 = y1_flipped + segment_end * unit_y
                                
                                # Draw this color segment
                                edge_pen = QPen(color)
                                edge_pen.setWidth(3)  # Slightly thicker for visibility
                                edge_pen.setStyle(Qt.SolidLine)
                                
                                line = QGraphicsLineItem(seg_x1, seg_y1, seg_x2, seg_y2)
                                line.setPen(edge_pen)
                                line.setZValue(1500)  # Draw above network but below GPS points
                                self.map_view.scene.addItem(line)
                                self._route_items.append(line)
                                
                                # Move to next segment
                                current_pos = segment_end
                                color_idx += 1
                        
                        drawn_edges.add(base_edge_id)
                    else:
                        # Draw edge with single color (solid line)
                        edge_pen = QPen(point_color)
                        edge_pen.setWidth(2)
                        edge_pen.setStyle(Qt.SolidLine)
                        
                        # Draw line segments along the edge shape
                        for i in range(len(shape) - 1):
                            x1, y1 = shape[i]
                            x2, y2 = shape[i + 1]
                            
                            # Flip Y coordinates
                            y1_flipped = flip_y(y1)
                            y2_flipped = flip_y(y2)
                            
                            line = QGraphicsLineItem(x1, y1_flipped, x2, y2_flipped)
                            line.setPen(edge_pen)
                            line.setZValue(1500)  # Draw above network but below GPS points
                            self.map_view.scene.addItem(line)
                            self._route_items.append(line)
                
                point_index += 1
        
        self.log(f"✅ Step 2: Drew {total_points} GPS points and their candidate edges (up to 10 regular edges within 300m + all Step 1 route edges) with distinct colors")
    
    def _draw_candidate_edges(self, candidate_edge_ids: List[str], color: QColor = None):
        """
        Draw candidate edges on the map in bright red (solid, not dashed).
        
        Args:
            candidate_edge_ids: List of edge IDs to draw
            color: QColor to use for drawing (default: bright red)
        """
        if not self.network_parser:
            return
        
        if color is None:
            color = QColor(255, 0, 0)  # Red
        
        edges_dict = self.network_parser.get_edges()
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPen
        from PySide6.QtWidgets import QGraphicsLineItem

        # Use bright red solid pen with thicker width for visibility
        candidate_pen = QPen(color)
        candidate_pen.setWidth(4)  # Thicker than route lines
        candidate_pen.setStyle(Qt.SolidLine)  # Solid line (not dashed)
        
        for edge_id in candidate_edge_ids:
            # Try base edge ID first, then with lane suffixes
            edge_data = None
            base_edge_id = edge_id.split('#')[0]  # Remove lane suffix if present
            
            if base_edge_id in edges_dict:
                edge_data = edges_dict[base_edge_id]
            else:
                # Try with lane suffix #0
                if f"{base_edge_id}#0" in edges_dict:
                    edge_data = edges_dict[f"{base_edge_id}#0"]
            
            if not edge_data:
                continue
            
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            # Use first lane's shape
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Draw line segments along the edge shape
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                
                # Flip Y coordinates
                y1_flipped = flip_y(y1)
                y2_flipped = flip_y(y2)
                
                line = QGraphicsLineItem(x1, y1_flipped, x2, y2_flipped)
                line.setPen(candidate_pen)
                line.setZValue(1500)  # Draw above route (1000) but below stars (2000)
                self.map_view.scene.addItem(line)
                self._route_items.append(line)
    
    def _draw_sumo_route_overlay(self, segments: List[List[List[float]]]):
        """
        Draw SUMO route overlay as dashed white lines using GPS-guided routing.
        Maps each GPS point to its nearest edge and builds route by following GPS trajectory.
        
        Args:
            segments: List of polyline segments (each segment is a list of [lon, lat] points)
        """
        if not self.network_parser:
            self.log("⚠️ Network parser not available")
            return
        
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPen
        from PySide6.QtWidgets import QGraphicsLineItem

        # Use cached sumolib network for gap filling (if available)
        net = self.sumo_net if hasattr(self, 'sumo_net') and self.sumo_net else None
        
        # Get Y bounds for flipping
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)
        
        def flip_y(y):
            """Flip Y coordinate to match network display orientation."""
            return y_max + y_min - y
        
        all_edges = []  # Collect all edges for logging
        
        # Process each segment separately (each is a new trip when split)
        for seg_idx, segment in enumerate(segments):
            if not segment or len(segment) < 2:
                continue
            
            # Get start and destination edges from first and last GPS points
            # (The new algorithm will find candidates in Step 1.1, but we need these for the fix)
            start_lon, start_lat = segment[0]
            dest_lon, dest_lat = segment[-1]
            
            # Find nearest edge for start GPS point
            start_sumo_coords = self.network_parser.gps_to_sumo_coords(start_lon, start_lat)
            if not start_sumo_coords:
                self.log(f"⚠️ Segment {seg_idx + 1}: Cannot convert start GPS point to SUMO coordinates")
                continue
            
            start_edge_result = self.network_parser.find_nearest_edge(
                start_sumo_coords[0], start_sumo_coords[1], allow_passenger_only=True
            )
            if not start_edge_result:
                self.log(f"⚠️ Segment {seg_idx + 1}: Cannot find nearest edge for start GPS point")
                continue
            
            start_edge_id = start_edge_result[0]
            
            # Find nearest edge for destination GPS point
            dest_sumo_coords = self.network_parser.gps_to_sumo_coords(dest_lon, dest_lat)
            if not dest_sumo_coords:
                self.log(f"⚠️ Segment {seg_idx + 1}: Cannot convert destination GPS point to SUMO coordinates")
                continue
            
            dest_edge_result = self.network_parser.find_nearest_edge(
                dest_sumo_coords[0], dest_sumo_coords[1], allow_passenger_only=True
            )
            if not dest_edge_result:
                self.log(f"⚠️ Segment {seg_idx + 1}: Cannot find nearest edge for destination GPS point")
                continue
            
            dest_edge_id = dest_edge_result[0]
            
            # Build route using Step 0 and Step 1 only (initial SUMO route calculation)
            all_routes = self._build_gps_validated_sumo_route(
                segment, start_edge_id, dest_edge_id, seg_idx
            )
            
            if not all_routes:
                self.log(f"⚠️ Segment {seg_idx + 1}: No valid route edges found")
                continue
            
            # Define distinct colors for different routes
            from PySide6.QtGui import QColor
            route_colors = [
                QColor(255, 200, 0),    # Fluorescent orange
                QColor(0, 255, 255),    # Cyan
                QColor(255, 0, 255),    # Magenta
                QColor(0, 255, 0),      # Green
                QColor(255, 165, 0),    # Orange
                QColor(255, 0, 128),    # Pink
                QColor(0, 128, 255),    # Blue
                QColor(128, 0, 255),    # Purple
                QColor(255, 128, 0),    # Red-orange
            ]
            
            # Log route information
            self.log(f"Segment {seg_idx + 1}: Found {len(all_routes)} route(s)")
            
            # Draw all routes with different colors
            edges_dict = self.network_parser.get_edges()
            for route_idx, route_edges in enumerate(all_routes):
                # Get color for this route (cycle through colors if more routes than colors)
                route_color = route_colors[route_idx % len(route_colors)]
                
                # Add edges to all_edges for summary
                all_edges.extend(route_edges)
                
                # Log route information
                edge_list_str = " → ".join(route_edges[:10])  # Show first 10 edges
                if len(route_edges) > 10:
                    edge_list_str += f" ... ({len(route_edges)} total)"
                self.log(f"  Route {route_idx + 1} ({len(route_edges)} edges): {edge_list_str}")
                
                # Draw dashed line along each edge in the route
                route_pen = QPen(route_color)
                route_pen.setWidth(3)
                route_pen.setStyle(Qt.DashLine)
                route_pen.setDashPattern([5, 5])  # 5px dash, 5px gap
                
                for edge_id in route_edges:
                    if edge_id not in edges_dict:
                        continue
                    
                    edge_data = edges_dict[edge_id]
                    lanes = edge_data.get('lanes', [])
                    if not lanes:
                        continue
                    
                    # Use first lane's shape
                    shape = lanes[0].get('shape', [])
                    if len(shape) < 2:
                        continue
                    
                    # Draw line segments along the edge shape
                    for i in range(len(shape) - 1):
                        x1, y1 = shape[i]
                        x2, y2 = shape[i + 1]
                        
                        # Flip Y coordinates
                        y1_flipped = flip_y(y1)
                        y2_flipped = flip_y(y2)
                        
                        line = QGraphicsLineItem(x1, y1_flipped, x2, y2_flipped)
                        line.setPen(route_pen)
                        line.setZValue(1000 + route_idx)  # Draw on top, with slight offset per route
                        self.map_view.scene.addItem(line)
                        self._route_items.append(line)
        
        # Log summary
        if all_edges:
            self.log(f"✅ SUMO route overlay (GPS-guided): {len(all_edges)} total edges mapped")
    
    def _find_parallel_edge(self, edge_id: str, edges_dict: dict) -> Optional[str]:
        """
        Find parallel edge (same junctions, opposite direction).
        
        Args:
            edge_id: Edge ID to find parallel for
            edges_dict: Dictionary of all edges
        
        Returns:
            Parallel edge ID or None if not found
        """
        if edge_id not in edges_dict:
            return None
        
        edge = edges_dict[edge_id]
        from_node = edge['from']
        to_node = edge['to']
        
        # Look for edge with reversed from/to nodes
        parallel_edge_id = f"{to_node}_{from_node}"  # Common SUMO naming pattern
        
        # Try different naming patterns
        candidates = [
            parallel_edge_id,
            f"{to_node}to{from_node}",
            f"{to_node}-{from_node}",
        ]
        
        # Also search by from/to nodes
        for candidate_id, candidate_edge in edges_dict.items():
            if (candidate_edge['from'] == to_node and 
                candidate_edge['to'] == from_node and
                candidate_id != edge_id):
                return candidate_id
        
        return None
    
    def _get_route_distance(self, from_edge_id: str, to_edge_id: str) -> Optional[float]:
        """
        Get shortest path distance between two edges using SUMO routing.
        
        Args:
            from_edge_id: Source edge ID
            to_edge_id: Destination edge ID
        
        Returns:
            Route distance in meters, or None if route not found
        """
        if not self.sumo_net:
            return None
        
        try:
            from_edge = self.sumo_net.getEdge(from_edge_id)
            to_edge = self.sumo_net.getEdge(to_edge_id)
            route_result = self.sumo_net.getShortestPath(from_edge, to_edge)
            
            if route_result and len(route_result) >= 2:
                edges, cost = route_result
                return cost  # Cost is typically distance in meters
            return None
        except Exception:
            return None
    
    def _get_base_edge_id(self, edge_id: str) -> str:
        """
        Get the base edge ID without the # suffix.
        
        Examples:
            '1016528107#1' -> '1016528107'
            '1016528107#0' -> '1016528107'
            '1016528107' -> '1016528107'
            '-1016528107#1' -> '-1016528107'
        
        Args:
            edge_id: Full edge ID (may include # suffix)
        
        Returns:
            Base edge ID without # suffix
        """
        if '#' in edge_id:
            return edge_id.split('#')[0]
        return edge_id
    
    def _find_candidate_edges_in_radius(
        self,
        lon: float,
        lat: float,
        max_candidates: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find candidate edges near GPS coordinates.
        
        Args:
            lon: Longitude
            lat: Latitude
            max_candidates: Maximum number of candidates to return
        
        Returns:
            List of (edge_id, distance) tuples sorted by distance
        """
        # TODO: Implement
        return []
    
    def _build_gps_validated_sumo_route(
        self,
        segment: List[List[float]],
        start_edge_id: str,
        dest_edge_id: str,
        seg_idx: int
    ) -> List[List[str]]:
        """
        Build route using SUMO routing (Step 0 and Step 1 only).
        
        Step 0: Spatial index is already prepared when network loads.
        Step 1: Calculate initial SUMO route from start to destination.
    
        Args:
            segment: GPS polyline segment (list of [lon, lat] points)
            start_edge_id: First edge ID (from first GPS point)
            dest_edge_id: Last edge ID (from last GPS point)
            seg_idx: Segment index for logging
    
        Returns:
            List of routes (each route is a list of edge IDs). Routes are sorted by similarity score (highest similarity first).
            The first route in the list has the highest similarity score and is the selected route.
        """
        if not self.sumo_net or not segment or len(segment) < 2:
            return []
        
        # Step 1: Calculate initial SUMO route (with iterative shortening)
        routes, start_candidates, dest_candidates, route_scores, final_segment = self._calculate_step1_route(segment, self.sumo_net, seg_idx)
        
        # Log similarity score of displayed route if available
        if route_scores and len(route_scores) > 0:
            best_score = route_scores[0]
            self.log(f"📍 Displayed SUMO route similarity score: {best_score:.4f} (Rank 1 - Best Match)")
        
        # Return all routes (sorted by similarity score, highest similarity first)
        # The first route is the selected one (highest similarity score)
        # Note: final_segment is returned but not used here (used in visualization)
        return routes if routes else []
    
    def _check_route_completeness(
        self, 
        segment: List[List[float]], 
        edges: List[str], 
        route_result
    ) -> bool:
        """
        Check if the SUMO route is complete from start to destination.
        
        Args:
            segment: GPS polyline segment
            edges: List of edge IDs in the route
            route_result: RouteMappingResult object
        
        Returns:
            True if route is complete, False otherwise
        """
        if not segment or len(segment) < 2 or not edges:
            return False
        
        if not route_result.edge_transitions:
            return False
        
        # Check 1: Route should start at the first GPS point's edge
        first_gps_point_edge = None
        first_transition = route_result.edge_transitions[0]
        if first_transition.gps_point_index == 0:
            first_gps_point_edge = first_transition.edge_id
        else:
            # First GPS point wasn't mapped, route is incomplete
            self.log(f"  ⚠️ First GPS point (index 0) was not mapped to an edge")
            return False
        
        if edges[0] != first_gps_point_edge:
            self.log(f"  ⚠️ Route does not start at first GPS point's edge. Expected: {first_gps_point_edge}, Got: {edges[0]}")
            return False
        
        # Check 2: Route should end at the last GPS point's edge
        last_gps_point_edge = None
        last_transition = route_result.edge_transitions[-1]
        if last_transition.gps_point_index == len(segment) - 1:
            last_gps_point_edge = last_transition.edge_id
        else:
            # Last GPS point wasn't mapped, route is incomplete
            self.log(f"  ⚠️ Last GPS point (index {len(segment) - 1}) was not mapped to an edge")
            return False
        
        if edges[-1] != last_gps_point_edge:
            self.log(f"  ⚠️ Route does not end at last GPS point's edge. Expected: {last_gps_point_edge}, Got: {edges[-1]}")
            return False
        
        # Check 3: All edges should be connected (each edge's 'to' node connects to next edge's 'from' node)
        edges_dict = self.network_parser.get_edges()
        for i in range(len(edges) - 1):
            current_edge_id = edges[i]
            next_edge_id = edges[i + 1]
            
            if current_edge_id not in edges_dict or next_edge_id not in edges_dict:
                self.log(f"  ⚠️ Edge not found in network: {current_edge_id if current_edge_id not in edges_dict else next_edge_id}")
                return False
            
            current_edge = edges_dict[current_edge_id]
            next_edge = edges_dict[next_edge_id]
            
            # Check if edges are connected
            if current_edge['to'] != next_edge['from']:
                self.log(f"  ⚠️ Edges not connected: {current_edge_id} (to: {current_edge['to']}) → {next_edge_id} (from: {next_edge['from']})")
                return False
        
        # Check 4: All GPS points should be mapped (no invalid points)
        if route_result.invalid_points:
            self.log(f"  ⚠️ {len(route_result.invalid_points)} GPS points could not be mapped to edges")
            return False
        
        return True
    
    def _on_draw_gps_points_path_changed(self):
        """Handle draw GPS points path checkbox state change."""
        self._schedule_save_settings()
        
        # If we have current segments, redraw the route (with or without GPS path)
        if self._current_segments is not None:
            self.show_selected_route()
        else:
            if not self._loading_settings and not self.draw_gps_points_path_checkbox.isChecked():
                # Clear route display if unchecked and route was displayed
                self.clear_route_display()
    
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
    
    def _draw_segment_candidate_edges(self, segment_polyline: List[List[float]]) -> None:
        """Draw green and orange candidate edges for a single trajectory segment."""
        if not self.network_parser or not segment_polyline or len(segment_polyline) < 2:
            return
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)

        def flip_y(y: float) -> float:
            return y_max + y_min - y

        sumo_points_flipped = []
        for lon, lat in segment_polyline:
            coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if coords:
                x, y = coords
                sumo_points_flipped.append((x, flip_y(y)))
        if len(sumo_points_flipped) < 2:
            return
        # Cache edges_data (expensive - iterates all network edges)
        if self._cached_edges_data is None:
            self._cached_edges_data = build_edges_data(self.network_parser)
        edges_data = self._cached_edges_data
        orange_ids, green_ids, _start, _end, _candidates = compute_green_orange_edges(
            edges_data, sumo_points_flipped, y_min, y_max, top_per_segment=5
        )
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        orange_pen = QPen(QColor(255, 165, 0), 5)
        orange_pen.setStyle(Qt.SolidLine)
        green_pen = QPen(QColor(0, 255, 0), 4)
        green_pen.setStyle(Qt.SolidLine)
        for eid in orange_ids:
            shape_points = edge_shapes.get(eid)
            if not shape_points or len(shape_points) < 2:
                continue
            for j in range(len(shape_points) - 1):
                x1, y1 = shape_points[j][0], shape_points[j][1]
                x2, y2 = shape_points[j + 1][0], shape_points[j + 1][1]
                line = self.map_view.scene.addLine(x1, flip_y(y1), x2, flip_y(y2), orange_pen)
                line.setZValue(21)
                self._candidate_edge_items.append(line)
        for eid in green_ids:
            shape_points = edge_shapes.get(eid)
            if not shape_points or len(shape_points) < 2:
                continue
            for j in range(len(shape_points) - 1):
                x1, y1 = shape_points[j][0], shape_points[j][1]
                x2, y2 = shape_points[j + 1][0], shape_points[j + 1][1]
                line = self.map_view.scene.addLine(x1, flip_y(y1), x2, flip_y(y2), green_pen)
                line.setZValue(20)
                self._candidate_edge_items.append(line)

    def _compute_edges_in_polygon(self, segments: List[List[List[float]]]) -> Optional[Set[str]]:
        """Compute set of edge IDs inside the trajectory polygon. Returns None if polygon cannot be built."""
        if not self.network_parser or not segments:
            return None
        all_gps = []
        for seg in segments:
            if seg and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 2:
                for pt in seg:
                    all_gps.append((pt[0], pt[1]))
        if len(all_gps) < 2:
            return None
        sumo_points_original = []
        for lon, lat in all_gps:
            coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if coords:
                sumo_points_original.append(coords)
        if len(sumo_points_original) < 2:
            return None
        min_box = self._find_minimum_bounding_box_route(sumo_points_original, padding_meters=200.0)
        if not min_box:
            return None
        center_x, center_y, width, height, angle = min_box
        half_w, half_h = width / 2, height / 2
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        rotated_corners = [QPointF(center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a) for dx, dy in corners]
        polygon_check = QPolygonF(rotated_corners)
        outside = [i for i, (x, y) in enumerate(sumo_points_original) if not polygon_check.containsPoint(QPointF(x, y), Qt.OddEvenFill)]
        if outside:
            center_x, center_y, width, height, angle = self._expand_box_to_include_all_points_route(
                sumo_points_original, center_x, center_y, width, height, angle, safety_margin=100.0
            )
            half_w, half_h = width / 2, height / 2
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            cos_a, sin_a = math.cos(-angle), math.sin(-angle)
            rotated_corners = [QPointF(center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a) for dx, dy in corners]
        if rotated_corners[0] != rotated_corners[-1]:
            rotated_corners.append(rotated_corners[0])
        polygon_original = QPolygonF(rotated_corners)
        if polygon_original.isEmpty():
            return None
        edges = self.network_parser.get_edges()
        edge_ids = set()
        for edge_id, edge_data in edges.items():
            if not edge_data.get('lanes'):
                continue
            shape_points = edge_data['lanes'][0].get('shape', [])
            if len(shape_points) < 2:
                continue
            edge_in_box = False
            for pt in shape_points:
                x, y = pt[0], pt[1]
                if polygon_original.containsPoint(QPointF(x, y), Qt.OddEvenFill):
                    edge_in_box = True
                    break
            if not edge_in_box:
                for i in range(len(shape_points) - 1):
                    x1, y1 = shape_points[i][0], shape_points[i][1]
                    x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
                    if self._line_intersects_polygon_route(QPointF(x1, y1), QPointF(x2, y2), polygon_original):
                        edge_in_box = True
                        break
            if edge_in_box:
                edge_ids.add(edge_id)
        return edge_ids

    MAX_SUMO_ROUTE_DISTANCE_M = 150.0  # Max distance from GPS point to route (reject if exceeded)

    def _segment_has_sumo_route(self, segment_polyline: List[List[float]], seg_idx: int = 0) -> Tuple[bool, str]:
        """Check if a SUMO route exists for this segment (view_network logic). Returns (True, '') if route found
        and all GPS points are within MAX_SUMO_ROUTE_DISTANCE_M of the route. Returns (False, reason) when rejected."""
        seg_label = f"Segment {seg_idx + 1}"
        if not self.network_parser or not segment_polyline:
            msg = "Network not loaded or segment empty"
            self.log(f"⚠️ {seg_label} SUMO route: not found ({msg})")
            return False, msg
        if len(segment_polyline) < 3:
            msg = "Too short (need at least 3 points)"
            self.log(f"⚠️ {seg_label} SUMO route: rejected ({msg})")
            return False, msg
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)

        def flip_y(y: float) -> float:
            return y_max + y_min - y

        sumo_points_flipped = []
        for lon, lat in segment_polyline:
            coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if coords:
                x, y = coords
                sumo_points_flipped.append((x, flip_y(y)))
        if len(sumo_points_flipped) < 2:
            msg = "Could not convert GPS points to SUMO coordinates"
            self.log(f"⚠️ {seg_label} SUMO route: not found ({msg})")
            return False, msg
        if self._cached_edges_data is None:
            self._cached_edges_data = build_edges_data(self.network_parser)
        edges_data = self._cached_edges_data
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        orange_ids, green_ids, start_id, end_id, candidates = compute_green_orange_edges(
            edges_data, sumo_points_flipped, y_min, y_max, top_per_segment=5
        )
        if not start_id or not end_id:
            msg = "No start or end edges for trajectory"
            self.log(f"⚠️ {seg_label} SUMO route: not found ({msg})")
            return False, msg
        node_positions = build_node_positions(self.network_parser)
        goal_xy = sumo_points_flipped[-1]
        base_edges = self._compute_edges_in_polygon(self._current_segments) if self._current_segments else None
        max_tries = min(5, len(candidates or []))
        max_star_distance_m = self.MAX_SUMO_ROUTE_DISTANCE_M
        worst_dist_overall = 0.0
        worst_idx_overall = 0
        for try_idx in range(max_tries):
            cand = (candidates or [])[try_idx]
            edges_allowed = (base_edges | {cand, end_id}) if base_edges is not None else None
            path_edges = shortest_path_dijkstra(
                self.network_parser,
                cand,
                end_id,
                orange_ids=orange_ids,
                green_ids=green_ids,
                node_positions=node_positions,
                goal_xy=goal_xy,
                edges_in_polygon=edges_allowed,
            )
            if not path_edges:
                continue
            # Build path points and validate 100m distance (route rejected if any point too far)
            path_points = []
            for eid in path_edges:
                shape_points = edge_shapes.get(eid)
                if not shape_points:
                    continue
                for x_s, y_s in shape_points:
                    path_points.append((x_s, flip_y(y_s)))
            path_points_flat = [[p[0], p[1]] for p in path_points]
            all_within = True
            worst_dist = 0.0
            worst_idx = 0
            for idx, (px, py) in enumerate(sumo_points_flipped):
                proj_x, proj_y = project_point_onto_polyline(px, py, path_points_flat)
                dist_m = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                if dist_m > max_star_distance_m:
                    all_within = False
                    if dist_m > worst_dist:
                        worst_dist = dist_m
                        worst_idx = idx
            if all_within:
                return True, ""
            if worst_dist > worst_dist_overall:
                worst_dist_overall = worst_dist
                worst_idx_overall = worst_idx
        if worst_dist_overall > max_star_distance_m:
            msg = f"GPS point {worst_idx_overall + 1} is {worst_dist_overall:.0f}m from route (max {max_star_distance_m:.0f}m)"
            self.log(f"⚠️ {seg_label} SUMO route: rejected ({msg})")
            return False, msg
        else:
            msg = "No path between start and end edges"
            self.log(f"⚠️ {seg_label} SUMO route: not found ({msg})")
            return False, msg

    def _draw_segment_sumo_route(self, segment_polyline: List[List[float]], seg_idx: int = 0) -> None:
        """Draw SUMO shortest path for a segment (view_network logic: Dijkstra/A*, purple path + stars)."""
        if not self.network_parser or not segment_polyline:
            return
        if len(segment_polyline) < 3:
            msg = "Too short (need at least 3 points)"
            self.log(f"⚠️ Segment {seg_idx + 1} SUMO route rejected: {msg}")
            if seg_idx < len(self._segment_sumo_route_status_labels):
                self._segment_sumo_route_status_labels[seg_idx].setText(msg)
                self._segment_sumo_route_status_labels[seg_idx].setStyleSheet("font-size: 9px; color: #c62828;")
            if seg_idx < len(self._segment_show_sumo_route_checkboxes):
                self._segment_show_sumo_route_checkboxes[seg_idx].blockSignals(True)
                self._segment_show_sumo_route_checkboxes[seg_idx].setChecked(False)
                self._segment_show_sumo_route_checkboxes[seg_idx].setEnabled(False)
                self._segment_show_sumo_route_checkboxes[seg_idx].blockSignals(False)
            return
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)

        def flip_y(y: float) -> float:
            return y_max + y_min - y

        sumo_points_flipped = []
        for lon, lat in segment_polyline:
            coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if coords:
                x, y = coords
                sumo_points_flipped.append((x, flip_y(y)))
        if len(sumo_points_flipped) < 2:
            return
        if self._cached_edges_data is None:
            self._cached_edges_data = build_edges_data(self.network_parser)
        edges_data = self._cached_edges_data
        edge_shapes = {eid: shape for eid, _ed, shape in edges_data}
        orange_ids, green_ids, start_id, end_id, candidates = compute_green_orange_edges(
            edges_data, sumo_points_flipped, y_min, y_max, top_per_segment=5
        )
        if not start_id or not end_id:
            return
        node_positions = build_node_positions(self.network_parser)
        goal_xy = sumo_points_flipped[-1]
        base_edges = self._compute_edges_in_polygon(self._current_segments) if self._current_segments else None
        max_tries = min(5, len(candidates or []))
        max_star_distance_m = self.MAX_SUMO_ROUTE_DISTANCE_M
        path_edges: List[str] = []
        worst_dist_overall = 0.0
        worst_idx_overall = 0
        for try_idx in range(max_tries):
            cand = (candidates or [])[try_idx]
            edges_allowed = (base_edges | {cand, end_id}) if base_edges is not None else None
            candidate_path = shortest_path_dijkstra(
                self.network_parser,
                cand,
                end_id,
                orange_ids=orange_ids,
                green_ids=green_ids,
                node_positions=node_positions,
                goal_xy=goal_xy,
                edges_in_polygon=edges_allowed,
            )
            if not candidate_path:
                continue
            path_points = []
            for eid in candidate_path:
                shape_points = edge_shapes.get(eid)
                if not shape_points:
                    continue
                for x_s, y_s in shape_points:
                    path_points.append((x_s, flip_y(y_s)))
            path_points_flat = [[p[0], p[1]] for p in path_points]
            all_within = True
            worst_dist = 0.0
            worst_idx = 0
            for idx, (px, py) in enumerate(sumo_points_flipped):
                proj_x, proj_y = project_point_onto_polyline(px, py, path_points_flat)
                dist_m = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                if dist_m > max_star_distance_m:
                    all_within = False
                    if dist_m > worst_dist:
                        worst_dist = dist_m
                        worst_idx = idx
            if all_within:
                path_edges = candidate_path
                break
            if worst_dist > worst_dist_overall:
                worst_dist_overall = worst_dist
                worst_idx_overall = worst_idx
        if not path_edges:
            if worst_dist_overall > 0:
                msg = f"GPS point {worst_idx_overall + 1} is {worst_dist_overall:.0f}m from route (max {max_star_distance_m:.0f}m)"
            else:
                msg = "No path between start and end edges"
            self.log(f"⚠️ Segment {seg_idx + 1} SUMO route rejected: {msg}")
            if seg_idx < len(self._segment_sumo_route_status_labels):
                self._segment_sumo_route_status_labels[seg_idx].setText(msg)
                self._segment_sumo_route_status_labels[seg_idx].setStyleSheet("font-size: 9px; color: #c62828;")
            if seg_idx < len(self._segment_show_sumo_route_checkboxes):
                self._segment_show_sumo_route_checkboxes[seg_idx].blockSignals(True)
                self._segment_show_sumo_route_checkboxes[seg_idx].setChecked(False)
                self._segment_show_sumo_route_checkboxes[seg_idx].setEnabled(False)
                self._segment_show_sumo_route_checkboxes[seg_idx].blockSignals(False)
            return
        # Build full path points (for projection and trimming)
        path_points = []
        for eid in path_edges:
            shape_points = edge_shapes.get(eid)
            if not shape_points:
                continue
            for x_s, y_s in shape_points:
                path_points.append((x_s, flip_y(y_s)))
        path_points_flat = [[p[0], p[1]] for p in path_points]
        # Trim route to start at first star and end at last star
        first_pt = sumo_points_flipped[0]
        last_pt = sumo_points_flipped[-1]
        (first_star_x, first_star_y), first_seg = project_point_onto_polyline_with_segment(
            first_pt[0], first_pt[1], path_points_flat
        )
        (last_star_x, last_star_y), last_seg = project_point_onto_polyline_with_segment(
            last_pt[0], last_pt[1], path_points_flat
        )
        if first_seg <= last_seg:
            # Build trimmed path: first_star -> intermediate points -> last_star
            trimmed = [(first_star_x, first_star_y)]
            for i in range(first_seg + 1, last_seg + 1):
                trimmed.append(path_points[i])
            if first_seg < last_seg:
                trimmed.append((last_star_x, last_star_y))
            elif first_seg == last_seg and (abs(first_star_x - last_star_x) > 1e-6 or abs(first_star_y - last_star_y) > 1e-6):
                trimmed.append((last_star_x, last_star_y))
            draw_points = trimmed
        else:
            # Degenerate: last star before first star along path; use full path
            draw_points = path_points
        # Draw the route polyline (from first star to last star)
        path_pen = QPen(QColor(128, 0, 128), 5)
        path_pen.setStyle(Qt.SolidLine)
        for j in range(len(draw_points) - 1):
            x1, y1 = draw_points[j][0], draw_points[j][1]
            x2, y2 = draw_points[j + 1][0], draw_points[j + 1][1]
            line = self.map_view.scene.addLine(x1, y1, x2, y2, path_pen)
            line.setZValue(25)
            self._sumo_route_items.append(line)
        if path_points and sumo_points_flipped:
            from PySide6.QtGui import QPainterPath
            path_points_flat = [[p[0], p[1]] for p in path_points]
            star_radius = 7.2
            star_brush = QBrush(QColor(255, 165, 0))
            star_pen = QPen(QColor(255, 165, 0), 1.8)
            font = QFont()
            font.setPointSize(7)
            font.setBold(True)
            red_number_color = QColor(200, 0, 0)
            for idx, (px, py) in enumerate(sumo_points_flipped):
                proj_x, proj_y = project_point_onto_polyline(px, py, path_points_flat)
                path = QPainterPath()
                num_points = 5
                outer, inner = star_radius, star_radius * 0.4
                for i in range(num_points * 2):
                    angle = (i * math.pi) / num_points - math.pi / 2
                    r = outer if i % 2 == 0 else inner
                    x = proj_x + r * math.cos(angle)
                    y = proj_y + r * math.sin(angle)
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                path.closeSubpath()
                from PySide6.QtWidgets import QGraphicsPathItem
                star_item = QGraphicsPathItem(path)
                star_item.setPen(star_pen)
                star_item.setBrush(star_brush)
                star_item.setZValue(30)
                self.map_view.scene.addItem(star_item)
                self._sumo_route_items.append(star_item)
                text_str = str(idx + 1)
                tx, ty = proj_x + star_radius, proj_y - star_radius
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    from PySide6.QtWidgets import QGraphicsTextItem
                    outline_item = QGraphicsTextItem(text_str)
                    outline_item.setDefaultTextColor(QColor(0, 0, 0))
                    outline_item.setFont(font)
                    outline_item.setPos(tx + dx * 0.5, ty + dy * 0.5)
                    outline_item.setZValue(31)
                    self.map_view.scene.addItem(outline_item)
                    self._sumo_route_items.append(outline_item)
                text_item = QGraphicsTextItem(text_str)
                text_item.setDefaultTextColor(red_number_color)
                text_item.setFont(font)
                text_item.setPos(tx, ty)
                text_item.setZValue(32)
                self.map_view.scene.addItem(text_item)
                self._sumo_route_items.append(text_item)

    def _draw_route_polygon_and_red_edges(self, segments: List[List[List[float]]], route_num: int = 0) -> None:
        """Draw polygon around trajectory (all segments) and color edges inside in red."""
        if not self.network_parser or not segments:
            return
        # Flatten segments to one list of GPS points and convert to original SUMO coordinates
        all_gps = []
        for seg in segments:
            if seg and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 2:
                for pt in seg:
                    all_gps.append((pt[0], pt[1]))
        if len(all_gps) < 2:
            return
        sumo_points_original = []
        for lon, lat in all_gps:
            coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if coords:
                sumo_points_original.append(coords)
        if len(sumo_points_original) < 2:
            return
        # #region agent log
        _log_path = "/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log"
        try:
            with open(_log_path, "a") as _f:
                _f.write(json.dumps({"hypothesisId":"B","location":"dataset_conversion_page:_draw_route_polygon","message":"Point counts","data":{"route_num":route_num,"all_gps":len(all_gps),"sumo_converted":len(sumo_points_original),"dropped":len(all_gps)-len(sumo_points_original)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        min_box = self._find_minimum_bounding_box_route(sumo_points_original, padding_meters=200.0)
        if not min_box:
            return
        center_x, center_y, width, height, angle = min_box
        half_w = width / 2
        half_h = height / 2
        corners = [
            (-half_w, -half_h), (half_w, -half_h),
            (half_w, half_h), (-half_w, half_h),
        ]
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rotated_corners = [QPointF(center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a) for dx, dy in corners]
        polygon_check = QPolygonF(rotated_corners)
        outside = [i for i, (x, y) in enumerate(sumo_points_original) if not polygon_check.containsPoint(QPointF(x, y), Qt.OddEvenFill)]
        if route_num == 495:
            try:
                with open(_log_path, "a") as _f:
                    _f.write(json.dumps({"hypothesisId":"D","location":"dataset_conversion_page:polygon_check","message":"Traj 495 initial containment","data":{"route_num":route_num,"outside_count":len(outside),"total_points":len(sumo_points_original),"outside_indices":outside[:15]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except Exception:
                pass
        if outside:
            # #region agent log
            try:
                with open(_log_path, "a") as _f:
                    _f.write(json.dumps({"hypothesisId":"A","location":"dataset_conversion_page:before_expand","message":"Points outside before expansion","data":{"route_num":route_num,"outside_count":len(outside),"outside_indices":outside[:10],"sample_outside":sumo_points_original[outside[0]] if outside else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except Exception:
                pass
            # #endregion
            center_x, center_y, width, height, angle = self._expand_box_to_include_all_points_route(
                sumo_points_original, center_x, center_y, width, height, angle, safety_margin=100.0
            )
            half_w, half_h = width / 2, height / 2
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            cos_a, sin_a = math.cos(-angle), math.sin(-angle)
            rotated_corners = [QPointF(center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a) for dx, dy in corners]
            # #region agent log
            polygon_after = QPolygonF(rotated_corners)
            still_outside = [i for i, (x, y) in enumerate(sumo_points_original) if not polygon_after.containsPoint(QPointF(x, y), Qt.OddEvenFill)]
            try:
                with open(_log_path, "a") as _f:
                    _f.write(json.dumps({"hypothesisId":"A","location":"dataset_conversion_page:after_expand","message":"Points outside after expansion","data":{"route_num":route_num,"still_outside_count":len(still_outside),"still_outside_indices":still_outside[:10],"sample_still_outside":sumo_points_original[still_outside[0]] if still_outside else None,"expand_result":{"cx":center_x,"cy":center_y,"w":width,"h":height}},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except Exception:
                pass
            # #endregion
        if rotated_corners[0] != rotated_corners[-1]:
            rotated_corners.append(rotated_corners[0])
        polygon_original = QPolygonF(rotated_corners)
        if polygon_original.isEmpty():
            return
        y_min = getattr(self.map_view, '_network_y_min', 0)
        y_max = getattr(self.map_view, '_network_y_max', 0)

        def flip_y(y):
            return y_max + y_min - y

        flipped_corners = [QPointF(c.x(), flip_y(c.y())) for c in rotated_corners]
        flipped_polygon = QPolygonF(flipped_corners)
        rect_item = QGraphicsPolygonItem(flipped_polygon)
        rect_item.setPen(QPen(QColor(0, 0, 0), 4))
        rect_item.setBrush(QBrush(Qt.NoBrush))
        rect_item.setZValue(50)
        self.map_view.scene.addItem(rect_item)
        self._route_items.append(rect_item)
        # Edges inside polygon (original SUMO coords)
        edges = self.network_parser.get_edges()
        red_pen = QPen(QColor(255, 180, 180, 140), 3)  # Pale red
        edges_in_box = []
        for edge_id, edge_data in edges.items():
            if not edge_data.get('lanes'):
                continue
            shape_points = edge_data['lanes'][0].get('shape', [])
            if len(shape_points) < 2:
                continue
            edge_in_box = False
            for pt in shape_points:
                x, y = pt[0], pt[1]
                if polygon_original.containsPoint(QPointF(x, y), Qt.OddEvenFill):
                    edge_in_box = True
                    break
            if not edge_in_box:
                for i in range(len(shape_points) - 1):
                    x1, y1 = shape_points[i][0], shape_points[i][1]
                    x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
                    if self._line_intersects_polygon_route(QPointF(x1, y1), QPointF(x2, y2), polygon_original):
                        edge_in_box = True
                        break
            if edge_in_box:
                edges_in_box.append((edge_id, edge_data, shape_points))
        for edge_id, edge_data, shape_points in edges_in_box:
            for i in range(len(shape_points) - 1):
                x1, y1 = shape_points[i][0], shape_points[i][1]
                x2, y2 = shape_points[i + 1][0], shape_points[i + 1][1]
                y1_d = flip_y(y1)
                y2_d = flip_y(y2)
                line = self.map_view.scene.addLine(x1, y1_d, x2, y2_d, red_pen)
                line.setZValue(15)
                self._route_items.append(line)
        self.log(f"✓ Polygon and {len(edges_in_box)} red edges drawn")

    def _find_minimum_bounding_box_route(self, points: List[Tuple[float, float]], padding_meters: float = 200.0) -> Optional[Tuple[float, float, float, float, float]]:
        """Minimum-area rotated bounding box. Returns (center_x, center_y, width, height, angle_rad) or None."""
        if len(points) < 2:
            return None
        padding = padding_meters
        min_area = float('inf')
        best_box = None
        for angle_deg in range(0, 181, 1):
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            cx = sum(x for x, y in points) / len(points)
            cy = sum(y for x, y in points) / len(points)
            rotated = [( (x - cx) * cos_a - (y - cy) * sin_a, (x - cx) * sin_a + (y - cy) * cos_a ) for x, y in points]
            xs, ys = [p[0] for p in rotated], [p[1] for p in rotated]
            w = max(xs) - min(xs) + 2 * padding
            h = max(ys) - min(ys) + 2 * padding
            if w * h < min_area:
                min_area = w * h
                best_box = (cx, cy, w, h, angle_rad)
        if best_box is None or best_box[2] <= 0 or best_box[3] <= 0:
            xs = [x for x, y in points]
            ys = [y for x, y in points]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            best_box = (cx, cy, max(xs) - min(xs) + 2 * padding, max(ys) - min(ys) + 2 * padding, 0.0)
        return best_box

    def _expand_box_to_include_all_points_route(self, points: List[Tuple[float, float]], center_x: float, center_y: float,
                                                width: float, height: float, angle: float, safety_margin: float = 50.0
                                                ) -> Tuple[float, float, float, float, float]:
        """Expand box so all points are inside. Returns (center_x, center_y, width, height, angle)."""
        if not points:
            return (center_x, center_y, width, height, angle)
        # Use same rotation as _find_minimum_bounding_box_route: R(angle) to transform world->local
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = []
        for x, y in points:
            dx, dy = x - center_x, y - center_y
            rotated.append((dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a))
        rxs, rys = [p[0] for p in rotated], [p[1] for p in rotated]
        min_rx, max_rx = min(rxs), max(rxs)
        min_ry, max_ry = min(rys), max(rys)
        req_w = (max_rx - min_rx) + 2 * safety_margin
        req_h = (max_ry - min_ry) + 2 * safety_margin
        center_rx = (min_rx + max_rx) / 2
        center_ry = (min_ry + max_ry) / 2
        # Transform center back to world: local = R(angle)*(world-center) => world = center + R(-angle)*local
        cos_neg, sin_neg = math.cos(-angle), math.sin(-angle)
        new_cx = center_x + center_rx * cos_neg - center_ry * sin_neg
        new_cy = center_y + center_rx * sin_neg + center_ry * cos_neg
        # #region agent log
        try:
            with open("/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId":"A","location":"dataset_conversion_page:_expand_box","message":"Expand computation","data":{"min_rx":min_rx,"max_rx":max_rx,"min_ry":min_ry,"max_ry":max_ry,"req_w":req_w,"req_h":req_h,"center_rx":center_rx,"center_ry":center_ry,"new_cx":new_cx,"new_cy":new_cy,"old_cx":center_x,"old_cy":center_y,"angle_deg":math.degrees(angle)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        return (new_cx, new_cy, max(width, req_w), max(height, req_h), angle)

    def _line_intersects_polygon_route(self, p1: QPointF, p2: QPointF, polygon: QPolygonF) -> bool:
        """True if segment p1-p2 intersects any edge of the polygon."""
        pts = [polygon.at(i) for i in range(polygon.size())]
        for i in range(len(pts)):
            p3, p4 = pts[i], pts[(i + 1) % len(pts)]
            if self._line_segments_intersect_route(p1, p2, p3, p4):
                return True
        return False

    def _line_segments_intersect_route(self, p1: QPointF, p2: QPointF, p3: QPointF, p4: QPointF) -> bool:
        def ccw(a, b, c):
            return (c.y() - a.y()) * (b.x() - a.x()) > (b.y() - a.y()) * (c.x() - a.x())
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4))

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
            
            # Apply GPS offset and convert to SUMO coordinates
            ox = self.map_offset_x_spinbox.value()
            oy = self.map_offset_y_spinbox.value()
            sumo_points = []
            for lon, lat in polyline:
                adj_lon, adj_lat = apply_gps_offset(lon, lat, ox, oy)
                result = self.network_parser.gps_to_sumo_coords(adj_lon, adj_lat)
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
            
            # Validate this segment to find invalid segments within it
            segment_validation = validate_trip_segments(polyline)
            segment_invalid_indices = set(segment_validation.invalid_segment_indices)
            
            # Draw lines connecting points
            for i in range(len(sumo_points) - 1):
                x1, y1 = sumo_points[i]
                x2, y2 = sumo_points[i + 1]
                
                # Check if this segment is invalid
                is_invalid = i in segment_invalid_indices
                
                # Also check original polyline invalid segments if showing single polyline
                if show_invalid_in_red and not is_multiple_segments and seg_idx == 0:
                    # For single polyline, also check original invalid segments
                    is_invalid = is_invalid or (i in invalid_segment_set)
                
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
            point_size = 30  # Half of original size (60 -> 30)
            font = QFont("Arial", 12, QFont.Bold)  # Smaller font for smaller circles
            
            # Draw points with numbers
            for i, (x, y) in enumerate(sumo_points):
                is_start = (i == 0)
                is_end = (i == len(sumo_points) - 1)
                
                # Outer circle: segment color for start/end, segment color for middle
                outer_color = segment_color
                outer_size = point_size
                
                # Create outer point circle
                outer_ellipse = QGraphicsEllipseItem(
                    x - outer_size/2, y - outer_size/2,
                    outer_size, outer_size
                )
                outer_ellipse.setBrush(QBrush(outer_color))
                outer_ellipse.setPen(QPen(QColor(255, 255, 255), 3))  # White border
                outer_ellipse.setZValue(10)  # On top
                self.map_view.scene.addItem(outer_ellipse)
                self._route_items.append(outer_ellipse)
                
                # Inner circle for start (green) and end (red) points
                if is_start or is_end:
                    inner_size = point_size * 0.75  # 75% of outer circle size for better visibility
                    if is_start:
                        inner_color = QColor(76, 175, 80)  # Green for start
                    else:
                        inner_color = QColor(244, 67, 54)  # Red for end
                    
                    inner_ellipse = QGraphicsEllipseItem(
                        x - inner_size/2, y - inner_size/2,
                        inner_size, inner_size
                    )
                    inner_ellipse.setBrush(QBrush(inner_color))
                    inner_ellipse.setPen(QPen(QColor(255, 255, 255), 2))  # White border
                    inner_ellipse.setZValue(11)  # Above outer circle
                    self.map_view.scene.addItem(inner_ellipse)
                    self._route_items.append(inner_ellipse)
                
                # Create point number text
                text = QGraphicsTextItem(str(point_counter))
                text.setFont(font)
                text.setDefaultTextColor(QColor(255, 255, 255))
                # Center text on point
                text_rect = text.boundingRect()
                text.setPos(x - text_rect.width()/2, y - text_rect.height()/2)
                text.setZValue(12)  # Above everything
                self.map_view.scene.addItem(text)
                self._route_items.append(text)
                
                point_counter += 1
        
        # Auto-fit view to route initially, then user can zoom freely
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
            # Reset zoom tracking after fitInView, but allow unlimited zooming after
            self.map_view.current_zoom = 1.0
        
        # Return validation result for UI updates (from first segment or combined)
        if segments:
            return validate_trip_segments(segments[0])
        return None
    
    def clear_route_display(self, hide_subsections: bool = False):
        """Clear the displayed route from the map.
        
        Args:
            hide_subsections: If True, hide the trajectory subsets section (e.g. when user clicks Clear).
        """
        if hasattr(self, '_route_items') and self._route_items:
            for item in self._route_items:
                self.map_view.scene.removeItem(item)
            self._route_items = []
            self.route_info_label.setText("Route cleared")
            if hasattr(self, 'invalid_segments_info_label'):
                self.invalid_segments_info_label.setText("")
            self.log("Route display cleared")
        if hasattr(self, '_candidate_edge_items') and self._candidate_edge_items:
            for item in self._candidate_edge_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    pass
            self._candidate_edge_items = []
        if hasattr(self, '_sumo_route_items') and self._sumo_route_items:
            for item in self._sumo_route_items:
                try:
                    self.map_view.scene.removeItem(item)
                except RuntimeError:
                    pass
            self._sumo_route_items = []
        
        if hide_subsections and hasattr(self, 'segment_subsections_group'):
            self.segment_subsections_group.setVisible(False)
        
        # Clear cached route data when route is cleared
        self._cached_route_num = None
        self._cached_original_polyline = None
        self._cached_route_data = None


