"""
SUMO Simulation page for running and monitoring traffic simulations.
"""

import traceback
from pathlib import Path
import json
import gzip
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QGroupBox, QTextEdit, QSlider, QCheckBox, QLineEdit, QFormLayout,
)
from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer, QRectF
from PySide6.QtGui import QFont, QColor, QBrush, QPainter

from src.gui.simulation_view import SimulationView
from src.utils.network_parser import NetworkParser
from src.utils.csv_to_steps_runner import create_dynamic_edges


def _format_sim_time_ww_dd_hh_mm_ss(seconds: float) -> str:
    """Format simulation time as WW:DD:HH:MM:SS (weeks, days, hours, minutes, seconds)."""
    s = max(0.0, float(seconds))
    weeks, s = divmod(s, 604800)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    secs = int(s)
    return f"{int(weeks):02d}:{int(days):02d}:{int(hours):02d}:{int(minutes):02d}:{secs:02d}"


def _parse_ww_dd_hh_mm_ss(s: str) -> int:
    """Parse WW:DD:HH:MM:SS to total seconds. Raises ValueError if invalid."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty time string")
    parts = s.split(":")
    if len(parts) != 5:
        raise ValueError("Expected WW:DD:HH:MM:SS (5 parts)")
    try:
        w, d, h, m, sec = [int(x) for x in parts]
    except ValueError:
        raise ValueError("All parts must be integers")
    if w < 0 or d < 0 or d > 6 or h < 0 or h > 23 or m < 0 or m > 59 or sec < 0 or sec > 59:
        raise ValueError("Out of range (weeks>=0, days 0-6, hours 0-23, min/sec 0-59)")
    return w * 604800 + d * 86400 + h * 3600 + m * 60 + sec
from src.utils.simulation_db_service import SimulationDBService
from src.utils.simulation_runner import SimulationRunner
from src.utils.sumo_config_manager import SUMOConfigManager
from src.utils.sumo_detector import auto_detect_sumo_home, effective_sumo_home, setup_sumo_environment


class MappingExportWorker(QObject):
    """Background worker for optional mapping file export."""

    progress = Signal(str)
    finished = Signal(bool, str, object)
    progress_counts = Signal(int, int)

    def __init__(self, project_name: str, project_path: str, output_folder: str):
        super().__init__()
        self.project_name = project_name
        self.project_path = project_path
        self.output_folder = output_folder

    def run(self):
        """Generate mapping files without blocking UI."""
        try:
            service = SimulationDBService(self.project_name, self.project_path)
            result = service.create_mapping_files_if_missing(
                self.output_folder,
                progress_cb=lambda msg: self.progress.emit(msg),
                progress_counts_cb=lambda c, t: self.progress_counts.emit(c, t),
            )
            self.finished.emit(True, "Mapping export finished.", result)
        except Exception as exc:
            self.finished.emit(False, str(exc), {})


class _SegmentedDatasetProgressBar(QWidget):
    """
    Single slim bar split into 3 segments:
    mapping / snapshots / labels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._map = (0, 1)
        self._snap = (0, 1)
        self._lab = (0, 1)

    def set_mapping(self, current: int, total: int):
        self._map = (max(0, int(current)), max(1, int(total)))
        self.update()

    def set_snapshots(self, current: int, total: int):
        self._snap = (max(0, int(current)), max(1, int(total)))
        self.update()

    def set_labels(self, current: int, total: int):
        self._lab = (max(0, int(current)), max(1, int(total)))
        self.update()

    def paintEvent(self, event):
        w = max(1, self.width())
        h = max(1, self.height())
        # Segment widths: Mapping 5%, Snapshots 85%, Labels 10%
        seg_fracs = (0.05, 0.85, 0.10)
        seg_w = [w * f for f in seg_fracs]

        def ratio(pair):
            c, t = pair
            return 0.0 if t <= 0 else max(0.0, min(1.0, float(c) / float(t)))

        r_map = ratio(self._map)
        r_snap = ratio(self._snap)
        r_lab = ratio(self._lab)

        # Light blue fill on a very light background.
        bg = QColor(235, 245, 255)
        fill = QColor(140, 195, 255)
        border = QColor(190, 215, 245)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(0.5, 0.5, w - 1.0, h - 1.0)
        radius = min(5.0, rect.height() / 2.0)
        p.setPen(border)
        p.setBrush(bg)
        p.drawRoundedRect(rect, radius, radius)

        def draw_segment(i: int, frac: float):
            x0 = rect.x() + sum(seg_w[:i])
            x1 = rect.x() + sum(seg_w[: i + 1])
            seg_rect = QRectF(x0, rect.y(), x1 - x0, rect.height())
            fill_rect = QRectF(seg_rect.x(), seg_rect.y(), seg_rect.width() * frac, seg_rect.height())
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(fill_rect, radius, radius)

            # Divider line between segments
            if i in (0, 1):
                p.setPen(border)
                p.drawLine(int(x1), int(rect.y()), int(x1), int(rect.y() + rect.height()))

        draw_segment(0, r_map)
        draw_segment(1, r_snap)
        draw_segment(2, r_lab)

        p.end()


class SimulationPage(QWidget):
    """Page for running and monitoring SUMO simulations."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, sumocfg_path: str, output_folder: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.sumocfg_path = sumocfg_path
        self.output_folder = output_folder
        self.config_manager = SUMOConfigManager(project_path)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with back button and title
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        back_btn = QPushButton("Back to Settings")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        header_layout.addWidget(back_btn)
        
        header_layout.addStretch()
        
        title = QLabel(f"SUMO Simulation - {self.project_name}")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Simulation Controls (Top)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(0, 0, 0, 10)
        
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_simulation)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_simulation)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        
        controls_layout.addStretch()
        
        # Step interval control (on same row as buttons)
        step_label = QLabel("Step Interval (ms):")
        step_label.setStyleSheet("color: #666; padding: 5px; font-size: 13px; font-weight: bold;")
        controls_layout.addWidget(step_label)
        
        # Step interval slider (0ms to 2000ms)
        # Step interval = real-world delay between rendering SUMO steps
        # 0ms = maximum speed (as fast as possible)
        # Lower interval = faster rendering (more steps per second)
        # Higher interval = slower rendering (fewer steps per second)
        self.step_interval_slider = QSlider(Qt.Horizontal)
        self.step_interval_slider.setMinimum(0)  # 0ms = maximum speed
        self.step_interval_slider.setMaximum(2000)  # 2000ms (0.5 steps/second = 0.5x speed if step=1s)
        self.step_interval_slider.setValue(0)  # Default: 0ms (maximum speed)
        self.step_interval_slider.setTickPosition(QSlider.TicksBelow)
        self.step_interval_slider.setTickInterval(200)
        self.step_interval_slider.setMaximumWidth(200)  # Limit width
        self.step_interval_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #f0f0f0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #45a049;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #45a049;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)
        self.step_interval_slider.valueChanged.connect(self.on_step_interval_changed)
        controls_layout.addWidget(self.step_interval_slider)
        
        # Step interval display label
        self.step_interval_label = QLabel("Max")
        self.step_interval_label.setStyleSheet("color: #666; padding: 5px 10px; font-size: 13px; font-weight: bold; min-width: 60px;")
        self.step_interval_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.step_interval_label)
        
        controls_layout.addStretch()
        
        # Simulation status
        status_label = QLabel("Status: Not Started")
        status_label.setStyleSheet("color: #666; padding: 5px 15px; font-size: 14px; font-weight: bold;")
        self.status_label = status_label
        controls_layout.addWidget(status_label)
        
        main_layout.addLayout(controls_layout)
        
        # Main simulation area: map on left, control buttons on right
        simulation_content_layout = QHBoxLayout()
        simulation_content_layout.setSpacing(10)

        # Left: map area (we grey it out via the container background in background runs).
        self.map_container = QWidget()
        map_container_layout = QVBoxLayout()
        map_container_layout.setContentsMargins(0, 0, 0, 0)
        map_container_layout.setSpacing(0)
        self.simulation_view = SimulationView()
        self.simulation_view.setMinimumHeight(400)
        map_container_layout.addWidget(self.simulation_view)
        self.map_container.setLayout(map_container_layout)
        simulation_content_layout.addWidget(self.map_container, stretch=1)

        # Right: simulation action buttons
        controls_group = QGroupBox("Simulation Controls")
        controls_group.setStyleSheet("""
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
        right_controls_layout = QVBoxLayout()
        right_controls_layout.setSpacing(10)
        right_controls_layout.setContentsMargins(10, 10, 10, 10)

        # Start/end time and run-in-background (saved/loaded from project JSON)
        self._sim_run_settings_path = Path(self.project_path) / "simulation_run_settings.json"
        try:
            sim_limit_sec = SimulationDBService(self.project_name, self.project_path).get_simulation_limit_seconds()
        except Exception:
            sim_limit_sec = 86400 + 1800
        default_end = _format_sim_time_ww_dd_hh_mm_ss(sim_limit_sec)
        settings = self._load_simulation_run_settings()
        start_time_str = settings.get("start_time", "00:00:00:00:00")
        end_time_str = settings.get("end_time", default_end)
        run_in_bg = bool(settings.get("run_in_background", False))
        compress_dataset = bool(settings.get("compress_dataset_output", False))

        time_form = QFormLayout()
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setPlaceholderText("00:00:00:00:00")
        self.start_time_edit.setText(start_time_str)
        self.start_time_edit.setToolTip("WW:DD:HH:MM:SS (weeks, days, hours, minutes, seconds). Max = simulation duration.")
        time_form.addRow("Start time:", self.start_time_edit)
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText(default_end)
        self.end_time_edit.setText(end_time_str)
        self.end_time_edit.setToolTip("WW:DD:HH:MM:SS. Must be after start time; max = simulation duration.")
        time_form.addRow("End time:", self.end_time_edit)
        right_controls_layout.addLayout(time_form)

        self.run_in_background_checkbox = QCheckBox("Run in background")
        self.run_in_background_checkbox.setToolTip("Grey out the map and skip rendering SUMO to the screen for faster runs.")
        self.run_in_background_checkbox.setChecked(run_in_bg)
        self.run_in_background_checkbox.toggled.connect(self.on_run_in_background_toggled)
        right_controls_layout.addWidget(self.run_in_background_checkbox)

        right_controls_layout.addWidget(self.start_btn)
        right_controls_layout.addWidget(self.pause_btn)
        right_controls_layout.addWidget(self.stop_btn)
        self.create_mappings_checkbox = QCheckBox("Create mapping files")
        self.create_mappings_checkbox.setToolTip(
            "If checked, create mapping files under <dataset_output>/mapping\n"
            "when starting simulation, only if that directory does not already exist."
        )
        right_controls_layout.addWidget(self.create_mappings_checkbox)
        
        # Controls future on-disk dataset export; must NOT disable dispatch/update or no vehicles
        # are injected from the simulation DB into SUMO.
        self.create_dataset_checkbox = QCheckBox("Create dataset")
        self.create_dataset_checkbox.setChecked(True)
        self.create_dataset_checkbox.setToolTip(
            "Reserved for writing dataset files after the run. "
            "Vehicles are always injected from the simulation DB (dispatch is always on)."
        )
        self.create_dataset_checkbox.toggled.connect(self._on_create_dataset_toggled)
        right_controls_layout.addWidget(self.create_dataset_checkbox)
        self.compress_dataset_checkbox = QCheckBox("Compress dataset files")
        self.compress_dataset_checkbox.setChecked(compress_dataset)
        self.compress_dataset_checkbox.setToolTip(
            "When dataset export is enabled, write compressed .json.gz files instead of .json."
        )
        self.compress_dataset_checkbox.toggled.connect(self._on_compress_dataset_toggled)
        right_controls_layout.addWidget(self.compress_dataset_checkbox)
        right_controls_layout.addStretch()
        controls_group.setLayout(right_controls_layout)
        controls_group.setMinimumWidth(220)
        controls_group.setMaximumWidth(280)
        simulation_content_layout.addWidget(controls_group, stretch=0)

        main_layout.addLayout(simulation_content_layout, stretch=1)

        # Simulation Log (Bottom)
        log_group = QGroupBox("Simulation Log")
        log_group.setStyleSheet("""
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
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)

        # Dataset progress UI (inside log area so it does not steal map height).
        self.dataset_progress_group = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(4, 2, 4, 4)
        progress_layout.setSpacing(4)
        legend = QLabel("Dataset Progress: Mapping / Snapshots / Labels")
        legend.setStyleSheet("color: #666; font-size: 11px;")
        progress_layout.addWidget(legend)
        self.dataset_progress_bar = _SegmentedDatasetProgressBar()
        self.dataset_progress_bar.setFixedHeight(10)
        progress_layout.addWidget(self.dataset_progress_bar)
        self.dataset_progress_group.setLayout(progress_layout)
        self.dataset_progress_group.setVisible(False)
        log_layout.addWidget(self.dataset_progress_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self.log_text.setPlaceholderText("Simulation log will appear here...")
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
        # Simulation state
        self.simulation_running = False
        self.simulation_paused = False
        self.run_in_background = bool(
            self.run_in_background_checkbox.isChecked()
            if hasattr(self, "run_in_background_checkbox")
            else False
        )
        self._apply_map_background(self.run_in_background)
        self.dataset_creation_enabled = bool(
            self.create_dataset_checkbox.isChecked()
            if hasattr(self, "create_dataset_checkbox")
            else True
        )
        self.compress_dataset_output = bool(
            self.compress_dataset_checkbox.isChecked()
            if hasattr(self, "compress_dataset_checkbox")
            else False
        )
        self.traci_connection = None
        self.network_parser = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_simulation)
        self.step_interval = 0  # Update interval in ms (default: 0ms = maximum speed)
        self.sumo_step_length = 1.0  # SUMO step length in seconds (default: 1.0)
        self._mapping_thread = None
        self._mapping_worker = None
        self._simulation_runner = None  # SimulationRunner for update/dispatch
        self._dataset_sampling_period_sec = 30
        self._dataset_snapshots_dir = None
        self._dataset_labels_dir = None
        self._dataset_snapshot_timestamps = []
        self._next_dataset_snapshot_sec = None
        self._dataset_static_written = False
        self._dataset_expected_snapshots_total = 0
        self._sim_end_sec_limit = None
        
        # Load network if sumocfg is available (after UI is complete)
        self.load_network()
        self.prepare_simulation_for_start()

    def prepare_simulation_for_start(self):
        """Validate DB readiness and clear roads/zones before allowing play."""
        self.start_btn.setEnabled(False)
        self.status_label.setText("Status: Preparing")
        self.status_label.setStyleSheet("color: #666; padding: 5px 15px; font-size: 14px; font-weight: bold;")
        self.log_text.append("Preparing simulation state...")
        try:
            service = SimulationDBService(self.project_name, self.project_path)
            ready, reason = service.validate_db_readiness_for_current_config()
            if not ready:
                self.log_text.append(f"Simulation DB not ready: {reason}")
                self.status_label.setText("Status: Blocked (DB not ready)")
                self.status_label.setStyleSheet(
                    "color: #f44336; padding: 5px 15px; font-size: 14px; font-weight: bold;"
                )
                return

            cleared = service.clear_roads_and_zones()
            self.log_text.append(
                f"Cleared roads/zones data: roads={cleared.get('roads_cleared', 0)}, "
                f"zones={cleared.get('zones_cleared', 0)}"
            )
            self.status_label.setText("Status: Ready")
            self.status_label.setStyleSheet(
                "color: #2e7d32; padding: 5px 15px; font-size: 14px; font-weight: bold;"
            )
            self.start_btn.setEnabled(True)
        except Exception as exc:
            self.log_text.append(f"Failed to prepare simulation state: {exc}")
            self.status_label.setText("Status: Blocked (prepare failed)")
            self.status_label.setStyleSheet(
                "color: #f44336; padding: 5px 15px; font-size: 14px; font-weight: bold;"
            )

    def _load_simulation_run_settings(self) -> dict:
        """Load run settings from project simulation_run_settings.json."""
        path = getattr(self, "_sim_run_settings_path", None) or Path(self.project_path) / "simulation_run_settings.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_simulation_run_settings(
        self,
        start_time: str,
        end_time: str,
        run_in_background: bool,
        compress_dataset_output: bool,
    ):
        """Save run settings to project simulation_run_settings.json."""
        path = getattr(self, "_sim_run_settings_path", None) or Path(self.project_path) / "simulation_run_settings.json"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "start_time": start_time,
                    "end_time": end_time,
                    "run_in_background": run_in_background,
                    "compress_dataset_output": compress_dataset_output,
                }, f, indent=2)
        except Exception:
            pass

    def _apply_map_background(self, run_in_background: bool) -> None:
        """Set the map background to greenish (normal) or reddish-grey (background run)."""
        try:
            # Matches SimulationView default background (QColor(195, 225, 195))
            normal_bg = QColor(195, 225, 195)
            # Redish-grey "blanked" background
            bg = QColor(176, 160, 160) if run_in_background else normal_bg
            brush = QBrush(bg)

            if hasattr(self, "simulation_view") and self.simulation_view is not None:
                # SimulationView uses QGraphicsScene background brush.
                try:
                    if hasattr(self.simulation_view, "scene") and self.simulation_view.scene is not None:
                        self.simulation_view.scene.setBackgroundBrush(brush)
                except Exception:
                    pass
                try:
                    self.simulation_view.setBackgroundBrush(brush)
                except Exception:
                    pass
        except Exception:
            pass

    def on_run_in_background_toggled(self, checked: bool):
        """Update UI/map appearance immediately when toggling background mode."""
        self.run_in_background = bool(checked)
        self._apply_map_background(self.run_in_background)

    def _on_create_dataset_toggled(self, checked: bool):
        """Mirror checkbox state for hooks that write dataset files (does not affect SUMO dispatch)."""
        self.dataset_creation_enabled = bool(checked)

    def _on_compress_dataset_toggled(self, checked: bool):
        """Mirror compression toggle state for dataset export hooks."""
        self.compress_dataset_output = bool(checked)

    def _json_load_safe(self, raw_value, default):
        if raw_value is None:
            return default
        try:
            value = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            return value if value is not None else default
        except Exception:
            return default

    def _fetch_junctions_for_dataset(self):
        with self._simulation_runner.sim_db.connect() as conn:
            rows = conn.execute(
                "SELECT id, x, y, type, zone, incoming_roads_json, outgoing_roads_json FROM junctions"
            ).fetchall()
        return [
            {
                "id": str(r[0]),
                "x": float(r[1] or 0.0),
                "y": float(r[2] or 0.0),
                "type": str(r[3] or ""),
                "zone": str(r[4] or ""),
                "incoming": self._json_load_safe(r[5], []),
                "outgoing": self._json_load_safe(r[6], []),
            }
            for r in rows
        ]

    def _fetch_road_edges_for_dataset(self):
        with self._simulation_runner.sim_db.connect() as conn:
            rows = conn.execute(
                "SELECT id, from_junction, to_junction, speed, length, num_lanes, zone, "
                "vehicles_on_road_json, density, avg_speed FROM roads"
            ).fetchall()
        roads = {}
        for r in rows:
            rid = str(r[0])
            roads[rid] = {
                "id": rid,
                "from": str(r[1] or ""),
                "to": str(r[2] or ""),
                "edge_type": 0,
                "speed": float(r[3] or 0.0),
                "length": float(r[4] or 0.0),
                "num_lanes": int(r[5] or 0),
                "zone": str(r[6] or ""),
                "vehicles_on_road": self._json_load_safe(r[7], []),
                "density": float(r[8] or 0.0),
                "avg_speed": float(r[9] or 0.0),
                "edge_demand": 0.0,
            }
        return roads

    def _fetch_active_nodes_for_dataset(self):
        with self._simulation_runner.sim_db.connect() as conn:
            rows = conn.execute(
                "SELECT id, vehicle_type, length, width, height, speed, acceleration, current_x, current_y, "
                "current_zone, current_edge, current_position, origin_name, origin_zone, origin_edge, "
                "origin_position, origin_x, origin_y, origin_start_sec, route_json, route_length, "
                "route_left_json, route_length_left, destination_name, destination_edge, destination_position, "
                "destination_x, destination_y FROM vehicles WHERE status = 'in_route'"
            ).fetchall()
        vehicles = {}
        for r in rows:
            vid = str(r[0])
            vehicles[vid] = {
                "id": vid,
                "vehicle_type": str(r[1] or ""),
                "length": float(r[2] or 0.0),
                "width": float(r[3] or 0.0),
                "height": float(r[4] or 0.0),
                "speed": float(r[5] or 0.0),
                "acceleration": float(r[6] or 0.0),
                "current_x": float(r[7] or 0.0),
                "current_y": float(r[8] or 0.0),
                "current_zone": str(r[9] or ""),
                "current_edge": str(r[10] or ""),
                "current_position": float(r[11] or 0.0),
                "origin_name": str(r[12] or ""),
                "origin_zone": str(r[13] or ""),
                "origin_edge": str(r[14] or ""),
                "origin_position": float(r[15] or 0.0),
                "origin_x": float(r[16] or 0.0),
                "origin_y": float(r[17] or 0.0),
                "origin_start_sec": int(r[18] or 0),
                "route": self._json_load_safe(r[19], []),
                "route_length": float(r[20] or 0.0),
                "route_left": self._json_load_safe(r[21], []),
                "route_length_left": float(r[22] or 0.0),
                "destination_name": str(r[23] or ""),
                "destination_edge": str(r[24] or ""),
                "destination_position": float(r[25] or 0.0),
                "destination_x": float(r[26] or 0.0),
                "destination_y": float(r[27] or 0.0),
            }
        return vehicles

    def _compute_edge_demand(self, road_edges: dict, vehicles: dict):
        for edge in road_edges.values():
            edge["edge_demand"] = 0.0
        tau_sec = 600.0
        for vehicle in vehicles.values():
            route_left = vehicle.get("route_left", [])
            if not route_left:
                continue
            current_edge = vehicle.get("current_edge", "")
            current_position = float(vehicle.get("current_position", 0.0) or 0.0)
            t = 0.0
            if current_edge:
                e = road_edges.get(current_edge, {})
                length = float(e.get("length", 0.0) or 0.0)
                speed = float(e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0)
                remaining = max(0.0, length - current_position)
                t += remaining / max(speed, 1e-6)
            for edge_id in route_left:
                e = road_edges.get(edge_id, {})
                if not e:
                    continue
                length = float(e.get("length", 0.0) or 0.0)
                speed = float(e.get("avg_speed", 0.0) or e.get("speed", 1.0) or 1.0)
                contribution = 1.0 / (1.0 + t / tau_sec)
                e["edge_demand"] = float(e.get("edge_demand", 0.0) or 0.0) + contribution
                t += length / max(speed, 1e-6)

    def _write_json_file(self, path: Path, payload: dict):
        if self.compress_dataset_output:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def _init_dataset_export(self):
        self._dataset_snapshot_timestamps = []
        self._dataset_snapshots_boundary_count = 0
        self._next_dataset_snapshot_sec = None
        self._dataset_static_written = False
        output_dir = Path(self.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_snapshots_dir = output_dir / "snapshots"
        self._dataset_labels_dir = output_dir / "labels"
        self._dataset_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_labels_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            total = int(self._dataset_expected_snapshots_total or 0)
            self.dataset_progress_bar.set_snapshots(0, total if total > 0 else 1)
            self.dataset_progress_bar.set_labels(0, 1)

    def _write_dataset_static_file(self):
        if not self.dataset_creation_enabled:
            return
        if self._simulation_runner is None:
            raise RuntimeError("Dataset static creation requires initialized SimulationRunner.")
        self._init_dataset_export()
        junctions = self._fetch_junctions_for_dataset()
        road_edges = self._fetch_road_edges_for_dataset()
        static_payload = {
            "junctions": junctions,
            "road_edges": [
                {
                    "id": e["id"],
                    "from": e["from"],
                    "to": e["to"],
                    "edge_type": e["edge_type"],
                    "speed": e["speed"],
                    "length": e["length"],
                    "num_lanes": e["num_lanes"],
                    "zone": e["zone"],
                }
                for e in road_edges.values()
            ],
        }
        static_name = "static.json.gz" if self.compress_dataset_output else "static.json"
        self._write_json_file(Path(self.output_folder) / static_name, static_payload)
        self._dataset_static_written = True
        self.log_text.append(f"Dataset static file created: {static_name}")
        if hasattr(self, "create_mappings_checkbox") and not self.create_mappings_checkbox.isChecked():
            if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
                self.dataset_progress_bar.set_mapping(1, 1)

    def _write_dataset_snapshot(self, snapshot_sec: int):
        if not self.dataset_creation_enabled or not self._dataset_static_written:
            return
        if self._simulation_runner is None or self._dataset_snapshots_dir is None:
            return
        self._dataset_snapshots_boundary_count = int(
            getattr(self, "_dataset_snapshots_boundary_count", 0)
        ) + 1
        total_exp = int(self._dataset_expected_snapshots_total or 0)
        denom = total_exp if total_exp > 0 else max(1, self._dataset_snapshots_boundary_count)

        road_edges = self._fetch_road_edges_for_dataset()
        vehicles = self._fetch_active_nodes_for_dataset()
        if not vehicles:
            if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
                self.dataset_progress_bar.set_snapshots(self._dataset_snapshots_boundary_count, denom)
            return
        self._compute_edge_demand(road_edges, vehicles)
        road_edges_dynamic = [
            {
                "id": e["id"],
                "vehicles_on_road": e.get("vehicles_on_road", []),
                "edge_demand": e.get("edge_demand", 0.0),
                "avg_speed": e.get("avg_speed", 0.0),
                "density": e.get("density", 0.0),
            }
            for e in road_edges.values()
            if e.get("vehicles_on_road")
            or e.get("edge_demand", 0.0) != 0.0
            or e.get("avg_speed", 0.0) != 0.0
            or e.get("density", 0.0) != 0.0
        ]
        step_payload = {
            "step": int(snapshot_sec),
            "nodes": list(vehicles.values()),
            "road_edges_dynamic": road_edges_dynamic,
            "dynamic_edges": create_dynamic_edges(road_edges, vehicles),
        }
        suffix = ".json.gz" if self.compress_dataset_output else ".json"
        path = self._dataset_snapshots_dir / f"step_{int(snapshot_sec):012d}{suffix}"
        self._write_json_file(path, step_payload)
        self._dataset_snapshot_timestamps.append(int(snapshot_sec))
        self.log_text.append(f"Dataset snapshot created: {path.name}")
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            self.dataset_progress_bar.set_snapshots(self._dataset_snapshots_boundary_count, denom)

    def _write_dataset_labels_after_run(self):
        if not self.dataset_creation_enabled or not self._dataset_snapshot_timestamps:
            return
        if self._simulation_runner is None or self._dataset_labels_dir is None:
            return
        with self._simulation_runner.sim_db.connect() as conn:
            rows = conn.execute(
                "SELECT id, destination_step FROM vehicles WHERE destination_step IS NOT NULL"
            ).fetchall()
        arrival_step_by_vehicle = {str(r[0]): int(r[1]) for r in rows if r[1] is not None}
        suffix = ".json.gz" if self.compress_dataset_output else ".json"
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            self.dataset_progress_bar.set_labels(0, len(self._dataset_snapshot_timestamps))
        created = 0
        for snapshot_sec in self._dataset_snapshot_timestamps:
            labels = []
            for vid, dest_step in arrival_step_by_vehicle.items():
                # Vehicles with eta==0 have completed their trip and should not appear
                # in snapshots or labels for this timestamp.
                if int(dest_step) <= int(snapshot_sec):
                    continue
                labels.append({"id": vid, "eta": int(dest_step) - int(snapshot_sec)})
            label_payload = {"timestamp": int(snapshot_sec), "labels": labels}
            path = self._dataset_labels_dir / f"label_{int(snapshot_sec):012d}{suffix}"
            self._write_json_file(path, label_payload)
            created += 1
            self.log_text.append(f"Dataset label created: {path.name}")
            if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
                self.dataset_progress_bar.set_labels(created, len(self._dataset_snapshot_timestamps))
            self.log_text.append(f"Dataset labels created for {len(self._dataset_snapshot_timestamps)} snapshot(s).")

    def _finalize_dataset_snapshot_progress(self):
        """Mark the snapshot segment full when the run ends (handles skipped empty snapshots and early stop)."""
        if not getattr(self, "dataset_creation_enabled", False):
            return
        if not hasattr(self, "dataset_progress_bar") or self.dataset_progress_bar is None:
            return
        n = int(getattr(self, "_dataset_snapshots_boundary_count", 0) or 0)
        if n <= 0:
            n = len(getattr(self, "_dataset_snapshot_timestamps", []) or [])
        if n <= 0:
            return
        self.dataset_progress_bar.set_snapshots(n, n)

    def load_network(self):
        """Load network from sumocfg file."""
        try:
            # Parse sumocfg to get network file
            import xml.etree.ElementTree as ET
            sumocfg_path = Path(self.sumocfg_path)
            if not sumocfg_path.exists():
                return
            
            tree = ET.parse(sumocfg_path)
            root = tree.getroot()
            input_elem = root.find('input')
            
            if input_elem is not None:
                # Try child element with value attribute
                net_elem = input_elem.find('net-file')
                if net_elem is not None:
                    net_file = net_elem.get('value')
                else:
                    # Try attribute directly
                    net_file = input_elem.get('net-file')
                
                if net_file:
                    # Resolve relative path
                    net_path = (sumocfg_path.parent / net_file).resolve()
                    if net_path.exists():
                        self.network_parser = NetworkParser(str(net_path))
                        self.simulation_view.load_network(self.network_parser)
                        self.log_text.append(f"Network loaded: {net_path}")
            
            # Get step length from sumocfg (default is 1.0 second)
            # SUMO step length = simulated time per step
            self.sumo_step_length = 1.0  # Default step length in seconds
            time_elem = root.find('time')
            if time_elem is not None:
                step_elem = time_elem.find('step-length')
                if step_elem is not None:
                    step_value = step_elem.get('value')
                    if step_value:
                        try:
                            self.sumo_step_length = float(step_value)
                        except ValueError:
                            pass
            
            # Step interval is the real-world delay between renders
            # Default: 1000ms = 1 render per second = 1 SUMO step per second = 1x speed (if step=1s)
            # Don't auto-adjust slider, let user control it
            
        except Exception as e:
            self.log_text.append(f"Error loading network: {str(e)}")
    
    def start_simulation(self):
        """Start the SUMO simulation."""
        if hasattr(self, "dataset_progress_group"):
            self.dataset_progress_group.setVisible(True)
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            self.dataset_progress_bar.set_mapping(0, 1)
            self.dataset_progress_bar.set_snapshots(0, 1)
            self.dataset_progress_bar.set_labels(0, 1)

        if hasattr(self, "create_mappings_checkbox") and self.create_mappings_checkbox.isChecked():
            if self._mapping_thread is not None:
                return
            self.log_text.append("Starting mapping export (background)...")
            if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
                self.dataset_progress_bar.set_mapping(0, 1)
            self.status_label.setText("Status: Preparing mappings")
            self.status_label.setStyleSheet("color: #666; padding: 5px 15px; font-size: 14px; font-weight: bold;")
            self.start_btn.setEnabled(False)
            self._start_mapping_export()
            return
        self._start_sumo_simulation()

    def _start_mapping_export(self):
        """Run optional mapping export asynchronously, then launch simulation."""
        self._mapping_thread = QThread(self)
        self._mapping_worker = MappingExportWorker(
            self.project_name,
            self.project_path,
            self.output_folder,
        )
        self._mapping_worker.moveToThread(self._mapping_thread)
        self._mapping_thread.started.connect(self._mapping_worker.run)
        self._mapping_worker.progress.connect(self.on_mapping_progress)
        self._mapping_worker.progress_counts.connect(self.on_mapping_counts)
        self._mapping_worker.finished.connect(self.on_mapping_finished)
        self._mapping_worker.finished.connect(self._mapping_thread.quit)
        self._mapping_thread.finished.connect(self._mapping_thread.deleteLater)
        self._mapping_thread.start()

    def on_mapping_progress(self, message: str):
        """Append mapping export progress in simulation log."""
        self.log_text.append(message)

    def on_mapping_counts(self, current: int, total: int):
        """Update mapping segment based on entity iteration progress."""
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            self.dataset_progress_bar.set_mapping(int(current), int(total))

    def on_mapping_finished(self, success: bool, message: str, result: object):
        """Continue startup flow after background mapping export."""
        if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
            self.dataset_progress_bar.set_mapping(1 if success else 0, 1)
        if success:
            if isinstance(result, dict):
                if result.get("created"):
                    self.log_text.append(
                        f"Mapping files created at {result.get('mapping_dir')}"
                    )
                else:
                    self.log_text.append(
                        f"Mapping directory already exists, skipped: {result.get('mapping_dir')}"
                    )
            self.log_text.append(message)
            self._start_sumo_simulation()
        else:
            self.log_text.append(f"Mapping export failed: {message}")
            self.status_label.setText("Status: Blocked (mapping failed)")
            self.status_label.setStyleSheet(
                "color: #f44336; padding: 5px 15px; font-size: 14px; font-weight: bold;"
            )
            if not self.simulation_running:
                self.start_btn.setEnabled(True)

        if self._mapping_worker is not None:
            self._mapping_worker.deleteLater()
        self._mapping_worker = None
        self._mapping_thread = None

    def _start_sumo_simulation(self):
        """Internal SUMO startup flow (called directly or after mapping export)."""
        self.log_text.append("Starting SUMO simulation...")
        self.log_text.append(f"SUMO Config: {self.sumocfg_path}")
        self.log_text.append(f"Output Folder: {self.output_folder}")
        self.log_text.append("")
        
        try:
            import os
            import sys

            # TraCI ships under $SUMO_HOME/tools (including PyPI eclipse-sumo). Configure path first.
            cfg_sumo_home = self.config_manager.get_sumo_home()
            sumo_home = effective_sumo_home(cfg_sumo_home or "") or None
            if not sumo_home:
                sumo_home = auto_detect_sumo_home()
            if sumo_home:
                setup_sumo_environment(sumo_home)
                if not cfg_sumo_home:
                    try:
                        self.config_manager.set_sumo_home(sumo_home)
                    except ValueError:
                        pass

            import traci
            
            # Try to import sumolib for binary checking
            try:
                import sumolib
                sumo_binary = sumolib.checkBinary('sumo')
            except ImportError:
                # Fallback: use 'sumo' directly or construct path from SUMO_HOME
                if sumo_home:
                    # Try to use SUMO binary from SUMO_HOME/bin
                    if os.name == 'nt':  # Windows
                        sumo_binary = os.path.join(sumo_home, 'bin', 'sumo.exe')
                    else:  # Linux/macOS
                        sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')
                    
                    if not os.path.exists(sumo_binary):
                        sumo_binary = 'sumo'  # Fallback to system PATH
                else:
                    sumo_binary = 'sumo'  # Fallback to system PATH
            
            # Simulation time range (user start/end, clamped to simulation duration)
            try:
                service = SimulationDBService(self.project_name, self.project_path)
                simulation_limit_sec = service.get_simulation_limit_seconds()
            except Exception:
                simulation_limit_sec = 86400 + 1800  # 1 day + 30 min fallback

            start_time_str = self.start_time_edit.text().strip() or "00:00:00:00:00"
            end_time_str = self.end_time_edit.text().strip()
            run_in_bg = self.run_in_background_checkbox.isChecked()
            compress_dataset_output = (
                self.compress_dataset_checkbox.isChecked()
                if hasattr(self, "compress_dataset_checkbox")
                else False
            )
            try:
                start_sec = _parse_ww_dd_hh_mm_ss(start_time_str)
                end_sec = _parse_ww_dd_hh_mm_ss(end_time_str) if end_time_str else simulation_limit_sec
            except ValueError as e:
                self.log_text.append(f"Invalid time format: {e}")
                QMessageBox.warning(
                    self,
                    "Invalid time",
                    f"Start and end must be in format WW:DD:HH:MM:SS.\n{e}"
                )
                return
            start_sec = max(0, min(start_sec, simulation_limit_sec))
            end_sec = max(0, min(end_sec, simulation_limit_sec))
            if start_sec >= end_sec:
                QMessageBox.warning(
                    self,
                    "Invalid range",
                    "Start time must be before end time."
                )
                return

            # Expected dataset snapshots count (used for progress bars)
            if self.dataset_creation_enabled:
                period = max(1, int(getattr(self, "_dataset_sampling_period_sec", 30) or 30))
                duration = max(0, int(end_sec) - int(start_sec))
                self._dataset_expected_snapshots_total = (duration // period) + 1
                if hasattr(self, "dataset_progress_bar") and self.dataset_progress_bar is not None:
                    t = max(1, int(self._dataset_expected_snapshots_total))
                    self.dataset_progress_bar.set_snapshots(0, t)
                    self.dataset_progress_bar.set_labels(0, t)
            self._save_simulation_run_settings(
                start_time_str,
                end_time_str,
                run_in_bg,
                compress_dataset_output,
            )
            self.run_in_background = run_in_bg
            self.compress_dataset_output = compress_dataset_output
            self._sim_end_sec_limit = int(end_sec)
            self.log_text.append(f"Simulation range: {_format_sim_time_ww_dd_hh_mm_ss(start_sec)} to {_format_sim_time_ww_dd_hh_mm_ss(end_sec)} (WW:DD:HH:MM:SS)")
            if run_in_bg:
                self.log_text.append("Running in background (map greyed out, no on-screen rendering).")

            # Start SUMO with user start/end.
            # SUMO's --start is a boolean flag; use --begin for numeric start time.
            sumo_cmd = [
                sumo_binary,
                '-c', self.sumocfg_path,
                '--begin', str(float(start_sec)),
                '--end', str(float(end_sec)),
                '--quit-on-end',  # Quit when simulation ends
            ]

            self._apply_map_background(run_in_bg)
            self._simulation_runner = SimulationRunner(self.project_path, self.project_name)
            if self.dataset_creation_enabled:
                self._write_dataset_static_file()

            traci.start(sumo_cmd)
            self.traci_connection = traci

            self.log_text.append("Simulation started successfully!")
            
            # Start update timer
            timer_interval = 1 if self.step_interval == 0 else self.step_interval
            self.update_timer.start(timer_interval)
            
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("color: #4CAF50; padding: 5px 15px; font-size: 14px; font-weight: bold;")
            
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.simulation_running = True
            self.simulation_paused = False
            
        except ImportError:
            self.log_text.append("ERROR: TraCI not available. Please ensure SUMO is installed and in PATH.")
            QMessageBox.warning(
                self,
                "SUMO Not Found",
                "SUMO or TraCI is not available. Please ensure SUMO is installed and accessible."
            )
        except Exception as e:
            self.log_text.append(f"ERROR: Failed to start simulation: {str(e)}")
            QMessageBox.warning(
                self,
                "Simulation Error",
                f"Failed to start simulation:\n{str(e)}"
            )
    
    def update_simulation(self):
        """Update simulation visualization.
        
        This is called every step_interval milliseconds.
        Each call advances the simulation by 1 SUMO step.
        Step interval controls the rendering speed:
        - Lower interval = more steps per second = faster simulation
        - Higher interval = fewer steps per second = slower simulation
        """
        if not self.simulation_running or not self.traci_connection:
            return
        
        try:
            import traci

            # 1. Advance SUMO by one step
            if not self.simulation_paused:
                traci.simulationStep()

            # Step that just completed (0 after first step), matching legacy loop index
            current_step = max(0, int(traci.simulation.getTime()) - 1)

            # 2. Update: sync DB from TraCI (vehicle positions, road occupancy, arrivals)
            if self._simulation_runner is not None:
                try:
                    self._simulation_runner.update(current_step, traci)
                except Exception as e:
                    self.log_text.append(f"Update error: {e}")

            # 3. Dispatch: add vehicles scheduled for this step to SUMO
            if self._simulation_runner is not None:
                try:
                    self._simulation_runner.dispatch(current_step, traci)
                except Exception as e:
                    self.log_text.append(f"Dispatch error: {e}")

            # 4. Periodic dataset snapshot export during run.
            if self.dataset_creation_enabled:
                sim_time_sec = int(traci.simulation.getTime())
                if self._next_dataset_snapshot_sec is None:
                    self._next_dataset_snapshot_sec = sim_time_sec
                while sim_time_sec >= int(self._next_dataset_snapshot_sec):
                    self._write_dataset_snapshot(int(self._next_dataset_snapshot_sec))
                    self._next_dataset_snapshot_sec += int(self._dataset_sampling_period_sec)

            # Get current sim time and vehicle count for status
            sim_time = traci.simulation.getTime()
            # Enforce configured end time (SUMO --end is not always reliable across setups).
            end_limit = getattr(self, "_sim_end_sec_limit", None)
            if end_limit is not None and sim_time >= float(end_limit):
                self.log_text.append("Reached configured end time. Stopping simulation and finalizing labels...")
                self.stop_simulation(confirm=False)
                return
            try:
                raw_ids = traci.vehicle.getIDList()
                vehicle_ids = [str(v) for v in (raw_ids if isinstance(raw_ids, (list, tuple)) else [])]
            except Exception:
                vehicle_ids = []
            vehicle_count = len(vehicle_ids)

            # Update vehicle positions on map only when not running in background (skip rendering)
            if not getattr(self, "run_in_background", False):
                current_vehicles = set()
                for vehicle_id in vehicle_ids:
                    try:
                        pos = traci.vehicle.getPosition(vehicle_id)
                        angle = traci.vehicle.getAngle(vehicle_id)
                        x = float(pos[0]) if len(pos) > 0 else 0.0
                        y = float(pos[1]) if len(pos) > 1 else 0.0
                        a = float(angle)
                        color_rgb = self._simulation_runner.get_vehicle_display_color(vehicle_id) if self._simulation_runner else None
                        self.simulation_view.update_vehicle(vehicle_id, x, y, a, color_rgb=color_rgb)
                        current_vehicles.add(vehicle_id)
                    except Exception:
                        pass
                existing_vehicles = set(self.simulation_view.vehicle_items.keys())
                for vehicle_id in existing_vehicles - current_vehicles:
                    self.simulation_view.remove_vehicle(vehicle_id)
            
            # Update status with speed information
            
            # Calculate effective speed: simulation seconds per real second
            # If step_interval = 0ms, then maximum speed
            # If step_interval = 500ms, then 2 steps/second = 2x speed (if step=1s)
            # If step_interval = 100ms, then 10 steps/second = 10x speed (if step=1s)
            if self.step_interval == 0:
                speed_text = "Max"
            else:
                steps_per_second = 1000.0 / self.step_interval
                effective_speed = steps_per_second * self.sumo_step_length  # simulation seconds per real second
                speed_text = f"{effective_speed:.2f}x"
            
            time_str = _format_sim_time_ww_dd_hh_mm_ss(sim_time)
            status_state = "Paused" if self.simulation_paused else "Running"
            self.status_label.setText(
                f"Status: {status_state} | Time: {time_str} | Vehicles: {vehicle_count} | "
                f"Speed: {speed_text}"
            )
            
        except Exception as e:
            self.log_text.append(f"Error updating simulation: {str(e)}")
            self.log_text.append(traceback.format_exc())
            self.stop_simulation()
    
    def pause_simulation(self):
        """Pause the SUMO simulation."""
        if self.simulation_paused:
            self.log_text.append("Resuming simulation...")
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("color: #4CAF50; padding: 10px; font-size: 14px; font-weight: bold;")
            self.pause_btn.setText("Pause")
            self.simulation_paused = False
        else:
            self.log_text.append("Pausing simulation...")
            self.status_label.setText("Status: Paused")
            self.status_label.setStyleSheet("color: #FF9800; padding: 10px; font-size: 14px; font-weight: bold;")
            self.pause_btn.setText("Resume")
            self.simulation_paused = True
        
        # TODO: Implement actual SUMO simulation pause/resume
    
    def stop_simulation(self, confirm: bool = True):
        """Stop the SUMO simulation."""
        if confirm and self.simulation_running:
            reply = QMessageBox.question(
                self,
                "Stop Simulation",
                "Are you sure you want to stop the simulation?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        self.log_text.append("Stopping simulation...")
        
        # Stop update timer
        self.update_timer.stop()
        
        # Close TraCI connection
        if self.traci_connection:
            try:
                import traci
                traci.close()
            except Exception:
                pass
            self.traci_connection = None

        if self.dataset_creation_enabled:
            try:
                self._finalize_dataset_snapshot_progress()
            except Exception:
                pass
            try:
                self._write_dataset_labels_after_run()
            except Exception as exc:
                self.log_text.append(f"Dataset label generation failed: {exc}")
        self._simulation_runner = None

        # Clear vehicles and restore normal map background.
        self.simulation_view.clear_vehicles()
        self.run_in_background = False
        self._apply_map_background(False)

        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #f44336; padding: 5px 15px; font-size: 14px; font-weight: bold;")
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self.simulation_running = False
        self.simulation_paused = False
        
        self.log_text.append("Simulation stopped.")
    
    def on_step_interval_changed(self, value: int):
        """Handle step interval slider change.
        
        Step interval = real-world delay between rendering SUMO steps.
        0ms = maximum speed (as fast as possible).
        Lower value = faster rendering (more steps per second).
        Higher value = slower rendering (fewer steps per second).
        """
        # Update step interval (in milliseconds)
        self.step_interval = value
        
        # Update display
        if value == 0:
            self.step_interval_label.setText("Max")
        else:
            self.step_interval_label.setText(f"{value} ms")
        
        # Calculate and show effective speed
        if hasattr(self, 'sumo_step_length'):
            if value == 0:
                # Maximum speed - show as "Max" or calculate theoretical max
                effective_speed = float('inf')
            else:
                steps_per_second = 1000.0 / value
                effective_speed = steps_per_second * self.sumo_step_length
            # Update status if simulation is running
            if self.simulation_running and hasattr(self, 'status_label'):
                # Status will be updated in next update_simulation call
                pass
        
        # Update timer interval if simulation is running
        if self.simulation_running and self.update_timer.isActive():
            self.update_timer.stop()
            if value == 0:
                # For 0ms, use 1ms as minimum (Qt timer minimum is 1ms)
                # This will run as fast as possible
                self.update_timer.start(1)
            else:
                self.update_timer.start(self.step_interval)

