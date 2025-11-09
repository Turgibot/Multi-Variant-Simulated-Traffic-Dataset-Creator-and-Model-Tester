"""
Dataset Generation page for creating traffic simulation datasets.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QGroupBox, QTextEdit, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from src.utils.sumo_config_manager import SUMOConfigManager
from src.utils.sumo_detector import auto_detect_sumo_home


class DatasetGenerationPage(QWidget):
    """Page for dataset generation."""
    
    back_clicked = Signal()
    run_simulation_clicked = Signal(str, str, str)  # Emits project_name, sumocfg_path, output_folder
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.config_manager = SUMOConfigManager(project_path)
        
        # Initialize validation state variables
        self.sumocfg_valid = False
        self.output_folder_valid = False
        self.sumo_home_valid = False
        
        self.init_ui()
        self.load_output_folder()
        self.load_sumo_home()
        # Load and display sumocfg contents if already set
        existing_sumocfg = self.config_manager.get_sumocfg_path()
        if existing_sumocfg:
            self.display_sumocfg_contents(existing_sumocfg)
    
    def init_ui(self):
        """Initialize the page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        
        back_btn = QPushButton("← Back to Projects")
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
        
        # Store existing sumocfg for validation after UI is complete
        existing_sumocfg = self.config_manager.get_sumocfg_path()
        if existing_sumocfg:
            self.sumocfg_input.setText(existing_sumocfg)
        
        # SUMO config file contents display
        self.config_contents_widget = QWidget()
        self.config_contents_layout = QVBoxLayout()
        self.config_contents_layout.setSpacing(5)
        self.config_contents_widget.setLayout(self.config_contents_layout)
        self.config_contents_widget.setVisible(False)
        config_layout.addWidget(self.config_contents_widget)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Dataset Configuration section
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_group.setStyleSheet("""
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
        dataset_layout = QVBoxLayout()
        dataset_layout.setSpacing(15)
        
        # Output folder selection
        output_folder_label = QLabel("Dataset Output Folder:")
        dataset_layout.addWidget(output_folder_label)
        
        output_folder_layout = QHBoxLayout()
        output_folder_layout.setSpacing(10)
        
        self.output_folder_input = QLineEdit()
        self.output_folder_input.setPlaceholderText("Enter path to output folder or use Browse...")
        self.output_folder_input.textChanged.connect(self.validate_output_folder)
        output_folder_layout.addWidget(self.output_folder_input)
        
        # Validation check mark
        self.output_folder_check = QLabel("✓")
        self.output_folder_check.setStyleSheet("color: #4CAF50; font-size: 18px; font-weight: bold;")
        self.output_folder_check.setVisible(False)
        output_folder_layout.addWidget(self.output_folder_check)
        
        # Browse button
        browse_folder_btn = QPushButton("Browse...")
        browse_folder_btn.setStyleSheet("""
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
        browse_folder_btn.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(browse_folder_btn)
        
        # Set button
        set_folder_btn = QPushButton("Set")
        set_folder_btn.setStyleSheet("""
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
        set_folder_btn.clicked.connect(self.set_output_folder)
        output_folder_layout.addWidget(set_folder_btn)
        
        dataset_layout.addLayout(output_folder_layout)
        
        # Info label
        info_label = QLabel(
            "Generated dataset files will be saved to the selected folder. "
            "If no folder is selected, datasets will be saved to the project's 'datasets' folder."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px; font-size: 11px;")
        dataset_layout.addWidget(info_label)
        
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)
        
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
        
        # SUMO_HOME path selection
        sumo_home_label = QLabel("SUMO_HOME Path:")
        sumo_config_layout.addWidget(sumo_home_label)
        
        sumo_home_layout = QHBoxLayout()
        sumo_home_layout.setSpacing(10)
        
        self.sumo_home_input = QLineEdit()
        self.sumo_home_input.setPlaceholderText("Enter path to SUMO installation directory or use Browse...")
        self.sumo_home_input.textChanged.connect(self.validate_sumo_home)
        sumo_home_layout.addWidget(self.sumo_home_input)
        
        # Validation check mark
        self.sumo_home_check = QLabel("✓")
        self.sumo_home_check.setStyleSheet("color: #4CAF50; font-size: 18px; font-weight: bold;")
        self.sumo_home_check.setVisible(False)
        sumo_home_layout.addWidget(self.sumo_home_check)
        
        # Browse button
        browse_sumo_home_btn = QPushButton("Browse...")
        browse_sumo_home_btn.setStyleSheet("""
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
        browse_sumo_home_btn.clicked.connect(self.browse_sumo_home)
        sumo_home_layout.addWidget(browse_sumo_home_btn)
        
        # Set button
        set_sumo_home_btn = QPushButton("Set")
        set_sumo_home_btn.setStyleSheet("""
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
        set_sumo_home_btn.clicked.connect(self.set_sumo_home)
        sumo_home_layout.addWidget(set_sumo_home_btn)
        
        sumo_config_layout.addLayout(sumo_home_layout)
        
        # Info label
        info_label = QLabel(
            "Set the path to your SUMO installation directory (SUMO_HOME). "
            "This is required for running SUMO simulations."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px; font-size: 11px;")
        sumo_config_layout.addWidget(info_label)
        
        sumo_config_group.setLayout(sumo_config_layout)
        main_layout.addWidget(sumo_config_group)
        
        # Run Simulation button (only enabled when all paths are valid)
        self.run_simulation_btn = QPushButton("▶ Run SUMO Simulation")
        self.run_simulation_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.run_simulation_btn.setEnabled(False)
        self.run_simulation_btn.clicked.connect(self.run_simulation)
        main_layout.addWidget(self.run_simulation_btn)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
        
        # Now that all UI elements are created, validate existing paths
        if existing_sumocfg:
            self.validate_sumocfg_path()
    
    def refresh_config_files(self):
        """Refresh the list of configuration files."""
        # Clear existing widgets
        while self.config_files_layout.count():
            item = self.config_files_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get all config files
        files = self.config_manager.get_config_files()
        
        if not files:
            # Show empty state
            empty_label = QLabel(
                "No SUMO configuration file loaded.\n"
                "Click 'Load SUMO Configuration File' above to load a .sumocfg file."
            )
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #999; padding: 20px;")
            self.config_files_layout.addWidget(empty_label)
        else:
            # Add file widgets
            for file_type, file_path in files.items():
                if isinstance(file_path, list):
                    # Multiple files of same type
                    for i, path in enumerate(file_path):
                        widget = SUMOConfigWidget(
                            f"{file_type} ({i+1})" if len(file_path) > 1 else file_type,
                            path
                        )
                        self.config_files_layout.addWidget(widget)
                else:
                    widget = SUMOConfigWidget(file_type, file_path)
                    self.config_files_layout.addWidget(widget)
        
        self.config_files_layout.addStretch()
    
    def load_output_folder(self):
        """Load the saved output folder path."""
        existing_folder = self.config_manager.get_dataset_output_folder()
        if existing_folder:
            self.output_folder_input.setText(existing_folder)
            self.validate_output_folder()
    
    def validate_sumocfg_path(self):
        """Validate the SUMO config file path."""
        path_text = self.sumocfg_input.text().strip()
        if not path_text:
            self.sumocfg_check.setVisible(False)
            self.sumocfg_valid = False
            self.update_run_button()
            return
        
        path = Path(path_text)
        is_valid = path.exists() and path.is_file() and path.suffix == '.sumocfg'
        
        self.sumocfg_check.setVisible(is_valid)
        self.sumocfg_valid = is_valid
        self.update_run_button()
    
    def validate_output_folder(self):
        """Validate the output folder path."""
        path_text = self.output_folder_input.text().strip()
        if not path_text:
            self.output_folder_check.setVisible(False)
            self.output_folder_valid = False
            self.update_run_button()
            return
        
        path = Path(path_text)
        is_valid = path.exists() and path.is_dir()
        
        self.output_folder_check.setVisible(is_valid)
        self.output_folder_valid = is_valid
        self.update_run_button()
    
    def update_run_button(self):
        """Update the Run Simulation button state based on validation."""
        # Check if button exists (may not be created yet during initialization)
        if not hasattr(self, 'run_simulation_btn'):
            return
        
        all_valid = self.sumocfg_valid and self.output_folder_valid and self.sumo_home_valid
        self.run_simulation_btn.setEnabled(all_valid)
    
    def browse_sumocfg(self):
        """Open file dialog to browse for SUMO config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SUMO Configuration File",
            str(Path.home()),
            "SUMO Config Files (*.sumocfg);;All Files (*)"
        )
        
        if file_path:
            self.sumocfg_input.setText(file_path)
            self.validate_sumocfg_path()
    
    def set_sumocfg_path(self):
        """Set the SUMO config file path from the input field."""
        path_text = self.sumocfg_input.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Error", "Please enter a path to the .sumocfg file.")
            return
        
        try:
            self.config_manager.set_sumocfg(path_text)
            self.display_sumocfg_contents(path_text)
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
            
            sumocfg_file = Path(sumocfg_path)
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
            
            # Title
            title_label = QLabel("Referenced Files:")
            title_font = QFont()
            title_font.setBold(True)
            title_label.setFont(title_font)
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
                            type_label.setStyleSheet("font-weight: bold; min-width: 120px;")
                            file_layout.addWidget(type_label)
                            
                            # File name
                            name_label = QLabel(file_name)
                            name_label.setStyleSheet("color: #666;")
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
                            type_label.setStyleSheet("font-weight: bold; min-width: 120px;")
                            file_layout.addWidget(type_label)
                            
                            # File name
                            name_label = QLabel(file_name)
                            name_label.setStyleSheet("color: #666;")
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
    
    def browse_output_folder(self):
        """Open file dialog to browse for output folder."""
        # Start from current folder if set, otherwise project datasets folder
        start_path = self.output_folder_input.text()
        if not start_path:
            start_path = str(Path(self.project_path) / 'datasets')
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Output Folder",
            start_path,
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            self.output_folder_input.setText(folder_path)
            self.validate_output_folder()
    
    def set_output_folder(self):
        """Set the output folder path from the input field."""
        path_text = self.output_folder_input.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Error", "Please enter a path to the output folder.")
            return
        
        try:
            # Save to JSON immediately
            self.config_manager.set_dataset_output_folder(path_text)
            QMessageBox.information(
                self,
                "Success",
                f"Dataset output folder set to:\n{path_text}"
            )
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def load_sumo_home(self):
        """Load the saved SUMO_HOME path or auto-detect."""
        existing_sumo_home = self.config_manager.get_sumo_home()
        if existing_sumo_home:
            self.sumo_home_input.setText(existing_sumo_home)
            self.validate_sumo_home()
        else:
            # Try to auto-detect SUMO_HOME
            detected_sumo_home = auto_detect_sumo_home()
            if detected_sumo_home:
                self.sumo_home_input.setText(detected_sumo_home)
                # Auto-save if detected
                try:
                    self.config_manager.set_sumo_home(detected_sumo_home)
                except ValueError:
                    pass  # If validation fails, don't save
                self.validate_sumo_home()
    
    def validate_sumo_home(self):
        """Validate the SUMO_HOME path."""
        path_text = self.sumo_home_input.text().strip()
        if not path_text:
            self.sumo_home_check.setVisible(False)
            self.sumo_home_valid = False
            self.update_run_button()
            return
        
        path = Path(path_text)
        # Check if path exists, is a directory, and has bin subdirectory
        is_valid = path.exists() and path.is_dir() and (path / 'bin').exists()
        
        self.sumo_home_check.setVisible(is_valid)
        self.sumo_home_valid = is_valid
        self.update_run_button()
    
    def browse_sumo_home(self):
        """Open file dialog to browse for SUMO_HOME directory."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select SUMO Installation Directory (SUMO_HOME)",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            self.sumo_home_input.setText(folder_path)
            self.validate_sumo_home()
    
    def set_sumo_home(self):
        """Set the SUMO_HOME path from the input field."""
        path_text = self.sumo_home_input.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Error", "Please enter a path to the SUMO installation directory.")
            return
        
        try:
            # Save to JSON immediately
            self.config_manager.set_sumo_home(path_text)
            QMessageBox.information(
                self,
                "Success",
                f"SUMO_HOME set to:\n{path_text}"
            )
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def run_simulation(self):
        """Run SUMO simulation - navigate to simulation page."""
        sumocfg_path = self.sumocfg_input.text().strip()
        output_folder = self.output_folder_input.text().strip()
        sumo_home = self.sumo_home_input.text().strip()
        
        if not sumocfg_path or not output_folder or not sumo_home:
            QMessageBox.warning(
                self,
                "Error",
                "Please set SUMO configuration file, output folder, and SUMO_HOME before running simulation."
            )
            return
        
        # Emit signal to navigate to simulation page
        self.run_simulation_clicked.emit(self.project_name, sumocfg_path, output_folder)

