"""
Dataset Generation page for creating traffic simulation datasets.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFileDialog, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QScrollArea, QTextEdit, QVBoxLayout, QWidget)

from src.utils.project_paths import resolve_path, to_display_path
from src.utils.sumo_config_manager import SUMOConfigManager
from src.utils.sumo_detector import effective_sumo_home


class DatasetGenerationPage(QWidget):
    """Page for dataset generation."""
    
    back_clicked = Signal()
    run_simulation_clicked = Signal(str, str, str)  # Emits project_name, sumocfg_path, output_folder
    route_generation_clicked = Signal()  # Emits when route generation is requested
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.config_manager = SUMOConfigManager(project_path)
        
        # Initialize validation state variables
        self.sumocfg_valid = False
        self.output_folder_valid = True
        self.sumo_home_valid = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        
        back_btn = QPushButton("Back to Projects")
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
        
        title = QLabel(f"Dataset Generation Settings - {self.project_name}")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Top-right action buttons
        header_actions_layout = QVBoxLayout()
        header_actions_layout.setSpacing(8)
        header_actions_layout.setContentsMargins(0, 0, 0, 0)

        # Run Simulation button
        self.run_simulation_btn = QPushButton("Run Simulation")
        self.run_simulation_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
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
        self.run_simulation_btn.setEnabled(False)
        self.run_simulation_btn.clicked.connect(self.run_simulation)
        header_actions_layout.addWidget(self.run_simulation_btn)

        define_simulation_btn = QPushButton("Define Simulation")
        define_simulation_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        define_simulation_btn.clicked.connect(self.route_generation_clicked.emit)
        header_actions_layout.addWidget(define_simulation_btn)
        header_layout.addLayout(header_actions_layout)
        
        main_layout.addLayout(header_layout)
        
        # SUMO Configuration section
        config_group = QGroupBox("SUMO Configuration Files")
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
        config_layout.setSpacing(10)
        
        # Info label
        info_label = QLabel(
            "Load a SUMO configuration file (.sumocfg) for this project. "
            "The system will automatically parse and find all referenced files "
            "(network, routes, additional files, etc.)."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px;")
        config_layout.addWidget(info_label)
        
        # SUMO config file path input
        sumocfg_label = QLabel("SUMO Configuration File (.sumocfg):")
        config_layout.addWidget(sumocfg_label)
        
        sumocfg_layout = QHBoxLayout()
        sumocfg_layout.setSpacing(10)
        
        self.sumocfg_input = QLineEdit()
        self.sumocfg_input.setPlaceholderText("Enter path to .sumocfg file or use Browse...")
        self.sumocfg_input.textChanged.connect(self.validate_sumocfg_path)
        sumocfg_layout.addWidget(self.sumocfg_input)
        
        # Validation check mark
        self.sumocfg_check = QLabel("✓")
        self.sumocfg_check.setStyleSheet("color: #4CAF50; font-size: 18px; font-weight: bold;")
        self.sumocfg_check.setVisible(False)
        sumocfg_layout.addWidget(self.sumocfg_check)
        
        # Browse button
        browse_sumocfg_btn = QPushButton("Browse...")
        browse_sumocfg_btn.setStyleSheet("""
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
        browse_sumocfg_btn.clicked.connect(self.browse_sumocfg)
        sumocfg_layout.addWidget(browse_sumocfg_btn)
        
        # Set button
        set_sumocfg_btn = QPushButton("Set")
        set_sumocfg_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        set_sumocfg_btn.clicked.connect(self.set_sumocfg_path)
        sumocfg_layout.addWidget(set_sumocfg_btn)
        
        config_layout.addLayout(sumocfg_layout)
        
        # SUMO config file contents display
        self.config_contents_widget = QWidget()
        self.config_contents_layout = QVBoxLayout()
        self.config_contents_layout.setSpacing(5)
        self.config_contents_widget.setLayout(self.config_contents_layout)
        self.config_contents_widget.setVisible(False)
        config_layout.addWidget(self.config_contents_widget)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # SUMO Configuration section
        sumo_config_group = QGroupBox("SUMO Configuration")
        sumo_config_group.setStyleSheet("""
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
        sumo_config_layout = QVBoxLayout()
        sumo_config_layout.setSpacing(15)
        
        self.sumo_status_label = QLabel("")
        self.sumo_status_label.setWordWrap(True)
        self.sumo_status_label.setStyleSheet(
            "color: #666; padding: 10px; font-size: 11px;"
        )
        sumo_config_layout.addWidget(self.sumo_status_label)

        hint_label = QLabel(
            "SUMO is located automatically (this environment's <b>eclipse-sumo</b> package, "
            "<tt>SUMO_HOME</tt>, <tt>sumo</tt> on <tt>PATH</tt>, or common install paths). "
            "Install for this project: <tt>uv pip install -r requirements.txt</tt> "
            "(includes <tt>eclipse-sumo</tt>)."
        )
        hint_label.setWordWrap(True)
        hint_label.setTextFormat(Qt.RichText)
        hint_label.setStyleSheet("color: #666; padding: 0 10px 10px; font-size: 11px;")
        sumo_config_layout.addWidget(hint_label)
        
        sumo_config_group.setLayout(sumo_config_layout)
        main_layout.addWidget(sumo_config_group)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
        
        # Load all saved settings after UI is fully initialized
        self.load_all_settings()
    
    def load_all_settings(self):
        """Load all saved project settings."""
        # Check if widgets exist (safety check)
        if not hasattr(self, 'sumocfg_input'):
            return
        
        # Temporarily block signals to avoid auto-save during loading
        self.sumocfg_input.blockSignals(True)
        
        try:
            # Load SUMO config file path
            existing_sumocfg = self.config_manager.get_sumocfg_path()
            if existing_sumocfg:
                self.sumocfg_input.setText(
                    to_display_path(existing_sumocfg, self.project_path)
                )
                self.validate_sumocfg_path()
                self.display_sumocfg_contents(existing_sumocfg)
            
            self.refresh_sumo_status()
        finally:
            # Re-enable signals and connect auto-save handlers
            self.sumocfg_input.blockSignals(False)
            
            # Connect auto-save handlers
            self.sumocfg_input.textChanged.connect(self.on_sumocfg_text_changed)
    
    def save_sumocfg_path(self, path_text: str):
        """Save SUMO config path to settings."""
        if path_text.strip():
            try:
                path = resolve_path(path_text.strip(), self.project_path)
                if path.exists() and path.suffix == '.sumocfg':
                    self.config_manager.set_sumocfg(path_text.strip())
            except Exception:
                pass  # Silently fail if path is invalid

    def on_sumocfg_text_changed(self):
        """Auto-save SUMO config path when text changes."""
        # Only save if validation passes (file exists and is valid)
        self.validate_sumocfg_path()
        if self.sumocfg_valid:
            self.save_sumocfg_path(self.sumocfg_input.text())
    
    def validate_sumocfg_path(self):
        """Validate the SUMO config file path."""
        path_text = self.sumocfg_input.text().strip()
        if not path_text:
            self.sumocfg_check.setVisible(False)
            self.sumocfg_valid = False
            self.update_run_button()
            return

        path = resolve_path(path_text, self.project_path)
        is_valid = path.exists() and path.is_file() and path.suffix == '.sumocfg'
        
        self.sumocfg_check.setVisible(is_valid)
        self.sumocfg_valid = is_valid
        self.update_run_button()
    
    def update_run_button(self):
        """Update the Run Simulation button state based on validation."""
        # Check if button exists (may not be created yet during initialization)
        if not hasattr(self, 'run_simulation_btn'):
            return
        
        all_valid = self.sumocfg_valid and self.sumo_home_valid
        self.run_simulation_btn.setEnabled(all_valid)
    
    def browse_sumocfg(self):
        """Open file dialog to browse for SUMO config file."""
        start = str(Path.home())
        cur = self.sumocfg_input.text().strip()
        if cur:
            p = resolve_path(cur, self.project_path)
            if p.exists():
                start = str(p.parent)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SUMO Configuration File",
            start,
            "SUMO Config Files (*.sumocfg);;All Files (*)"
        )

        if file_path:
            self.sumocfg_input.setText(
                to_display_path(file_path, self.project_path)
            )
            self.validate_sumocfg_path()
    
    def set_sumocfg_path(self):
        """Set the SUMO config file path from the input field."""
        path_text = self.sumocfg_input.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Error", "Please enter a path to the .sumocfg file.")
            return
        
        try:
            self.config_manager.set_sumocfg(path_text)
            abs_cfg = str(resolve_path(path_text, self.project_path))
            self.display_sumocfg_contents(abs_cfg)
            self.sumocfg_input.setText(to_display_path(abs_cfg, self.project_path))
            QMessageBox.information(
                self,
                "Success",
                f"SUMO configuration file set successfully!\n\n"
                f"Found {len(self.config_manager.get_config_files()) - 1} referenced files."
            )
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to load SUMO configuration file:\n{str(e)}"
            )
    
    def display_sumocfg_contents(self, sumocfg_path: str):
        """Display the contents of the sumocfg file."""
        # Clear existing content
        while self.config_contents_layout.count():
            item = self.config_contents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        try:
            import xml.etree.ElementTree as ET
            from pathlib import Path
            
            sumocfg_file = resolve_path(sumocfg_path, self.project_path)
            if not sumocfg_file.exists():
                return

            sumocfg_dir = sumocfg_file.parent
            tree = ET.parse(sumocfg_file)
            root = tree.getroot()
            
            # Find input element
            input_elem = root.find('input')
            if input_elem is None:
                self.config_contents_widget.setVisible(False)
                return
            
            # Title (explicit dark text so theme palette does not wash it out)
            title_label = QLabel("Referenced Files:")
            title_font = QFont()
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setStyleSheet("color: #212121;")
            self.config_contents_layout.addWidget(title_label)
            
            # Display each file type
            # SUMO config can have attributes directly on <input> or child elements with value attribute
            file_types = [
                ('net-file', 'Network File'),
                ('route-files', 'Route Files'),
                ('additional-files', 'Additional Files'),
                ('configuration-file', 'Configuration File')
            ]
            
            has_files = False
            
            # First try: child elements with value attribute (e.g., <net-file value="..."/>)
            for elem_name, display_name in file_types:
                elem = input_elem.find(elem_name)
                if elem is not None:
                    file_value = elem.get('value')
                    if file_value:
                        has_files = True
                        # Handle multiple files (comma-separated)
                        files = [f.strip() for f in file_value.split(',') if f.strip()]
                        
                        for file_name in files:
                            file_path = (sumocfg_dir / file_name).resolve()
                            file_exists = file_path.exists()
                            
                            # Create file info widget
                            file_layout = QHBoxLayout()
                            file_layout.setSpacing(10)
                            
                            # File type label
                            type_label = QLabel(f"{display_name}:")
                            type_label.setStyleSheet(
                                "font-weight: bold; min-width: 120px; color: #212121;"
                            )
                            file_layout.addWidget(type_label)
                            
                            # File name
                            name_label = QLabel(file_name)
                            name_label.setStyleSheet("color: #212121;")
                            file_layout.addWidget(name_label)
                            
                            file_layout.addStretch()
                            
                            # Existence check
                            if file_exists:
                                status_label = QLabel("✓")
                                status_label.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
                            else:
                                status_label = QLabel("✗")
                                status_label.setStyleSheet("color: #f44336; font-size: 16px; font-weight: bold;")
                                status_label.setToolTip(f"File not found: {file_path}")
                            
                            file_layout.addWidget(status_label)
                            
                            # Container widget
                            file_widget = QWidget()
                            file_widget.setLayout(file_layout)
                            file_widget.setStyleSheet("""
                                QWidget {
                                    background-color: #f9f9f9;
                                    border: 1px solid #ddd;
                                    border-radius: 3px;
                                    padding: 5px;
                                }
                            """)
                            self.config_contents_layout.addWidget(file_widget)
            
            # Second try: attributes directly on <input> element (fallback)
            if not has_files:
                for attr_name, display_name in file_types:
                    file_value = input_elem.get(attr_name)
                    if file_value:
                        has_files = True
                        # Handle multiple files (comma-separated)
                        files = [f.strip() for f in file_value.split(',') if f.strip()]
                        
                        for file_name in files:
                            file_path = (sumocfg_dir / file_name).resolve()
                            file_exists = file_path.exists()
                            
                            # Create file info widget
                            file_layout = QHBoxLayout()
                            file_layout.setSpacing(10)
                            
                            # File type label
                            type_label = QLabel(f"{display_name}:")
                            type_label.setStyleSheet(
                                "font-weight: bold; min-width: 120px; color: #212121;"
                            )
                            file_layout.addWidget(type_label)
                            
                            # File name
                            name_label = QLabel(file_name)
                            name_label.setStyleSheet("color: #212121;")
                            file_layout.addWidget(name_label)
                            
                            file_layout.addStretch()
                            
                            # Existence check
                            if file_exists:
                                status_label = QLabel("✓")
                                status_label.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
                            else:
                                status_label = QLabel("✗")
                                status_label.setStyleSheet("color: #f44336; font-size: 16px; font-weight: bold;")
                                status_label.setToolTip(f"File not found: {file_path}")
                            
                            file_layout.addWidget(status_label)
                            
                            # Container widget
                            file_widget = QWidget()
                            file_widget.setLayout(file_layout)
                            file_widget.setStyleSheet("""
                                QWidget {
                                    background-color: #f9f9f9;
                                    border: 1px solid #ddd;
                                    border-radius: 3px;
                                    padding: 5px;
                                }
                            """)
                            self.config_contents_layout.addWidget(file_widget)
            
            if has_files:
                self.config_contents_widget.setVisible(True)
            else:
                self.config_contents_widget.setVisible(False)
                
        except Exception as e:
            # If parsing fails, just hide the widget
            self.config_contents_widget.setVisible(False)
    
    def refresh_sumo_status(self):
        """Update SUMO detection label and run-button gating from config + auto-detect."""
        cfg = self.config_manager.get_sumo_home()
        eh = effective_sumo_home(cfg or "")
        if not eh:
            self.sumo_status_label.setText(
                "SUMO not found. Run: uv pip install -r requirements.txt "
                "(installs eclipse-sumo), or set SUMO_HOME / install SUMO on PATH."
            )
            self.sumo_status_label.setStyleSheet(
                "color: #f44336; padding: 10px; font-size: 11px;"
            )
            self.sumo_home_valid = False
            self.update_run_button()
            return

        self.sumo_status_label.setText(
            f"SUMO ready: {to_display_path(eh, self.project_path)}"
        )
        self.sumo_status_label.setStyleSheet(
            "color: #4CAF50; padding: 10px; font-size: 11px;"
        )
        self.sumo_home_valid = True
        self.update_run_button()
    
    def run_simulation(self):
        """Run SUMO simulation - navigate to simulation page."""
        sumocfg_path = str(resolve_path(self.sumocfg_input.text().strip(), self.project_path))
        output_folder = self.config_manager.get_dataset_output_folder()
        if not output_folder:
            output_folder = str((Path(self.project_path) / "datasets").resolve())
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            try:
                self.config_manager.set_dataset_output_folder(output_folder)
            except Exception:
                pass

        eh = effective_sumo_home(self.config_manager.get_sumo_home() or "")
        if not sumocfg_path or not eh:
            QMessageBox.warning(
                self,
                "Error",
                "Please set a valid .sumocfg file. SUMO could not be located automatically.",
            )
            return
        
        # Emit signal to navigate to simulation page
        self.run_simulation_clicked.emit(self.project_name, sumocfg_path, output_folder)

