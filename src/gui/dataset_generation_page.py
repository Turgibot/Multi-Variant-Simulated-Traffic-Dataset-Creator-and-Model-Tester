"""
Dataset Generation page for creating traffic simulation datasets.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QGroupBox, QTextEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from src.utils.sumo_config_manager import SUMOConfigManager


class SUMOConfigWidget(QFrame):
    """Widget for displaying and managing a SUMO configuration file."""
    
    file_removed = Signal(str)  # Emits file_type
    
    def __init__(self, file_type: str, file_path: str, parent=None):
        super().__init__(parent)
        self.file_type = file_type
        self.file_path = file_path
        self.init_ui()
    
    def init_ui(self):
        """Initialize the widget UI."""
        self.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        
        # File info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        type_label = QLabel(self.file_type.upper())
        type_font = QFont()
        type_font.setBold(True)
        type_label.setFont(type_font)
        info_layout.addWidget(type_label)
        
        path_label = QLabel(self.file_path)
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        path_label.setWordWrap(True)
        info_layout.addWidget(path_label)
        
        # Check if file exists
        if not Path(self.file_path).exists():
            status_label = QLabel("⚠ File not found")
            status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            info_layout.addWidget(status_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        remove_btn.clicked.connect(lambda: self.file_removed.emit(self.file_type))
        layout.addWidget(remove_btn)
        
        self.setLayout(layout)


class DatasetGenerationPage(QWidget):
    """Page for dataset generation."""
    
    back_clicked = Signal()
    
    def __init__(self, project_name: str, project_path: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.project_path = project_path
        self.config_manager = SUMOConfigManager(project_path)
        self.init_ui()
        self.refresh_config_files()
    
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
        
        title = QLabel(f"Dataset Generation - {self.project_name}")
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
        
        # Load SUMO config button
        load_config_btn = QPushButton("+ Load SUMO Configuration File (.sumocfg)")
        load_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        load_config_btn.clicked.connect(self.load_sumocfg)
        config_layout.addWidget(load_config_btn)
        
        # Config files scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        self.config_files_widget = QWidget()
        self.config_files_layout = QVBoxLayout()
        self.config_files_layout.setSpacing(10)
        self.config_files_widget.setLayout(self.config_files_layout)
        scroll.setWidget(self.config_files_widget)
        config_layout.addWidget(scroll)
        
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Dataset Configuration section (placeholder)
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
        
        placeholder_label = QLabel("Dataset configuration options will be available here.")
        placeholder_label.setStyleSheet("color: #999; padding: 20px;")
        placeholder_label.setAlignment(Qt.AlignCenter)
        dataset_layout.addWidget(placeholder_label)
        
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
    
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
                        widget.file_removed.connect(self.remove_sumocfg)
                        self.config_files_layout.addWidget(widget)
                else:
                    widget = SUMOConfigWidget(file_type, file_path)
                    if file_type == 'sumocfg':
                        widget.file_removed.connect(self.remove_sumocfg)
                    else:
                        # Individual files can't be removed, only the main sumocfg
                        widget.setEnabled(False)
                        # Hide remove button for parsed files
                        for child in widget.findChildren(QPushButton):
                            if child.text() == "Remove":
                                child.hide()
                    self.config_files_layout.addWidget(widget)
        
        self.config_files_layout.addStretch()
    
    def load_sumocfg(self):
        """Load a SUMO configuration file (.sumocfg)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SUMO Configuration File",
            str(Path.home()),
            "SUMO Config Files (*.sumocfg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.config_manager.set_sumocfg(file_path)
                QMessageBox.information(
                    self,
                    "Success",
                    f"SUMO configuration file loaded successfully!\n\n"
                    f"Found {len(self.config_manager.get_config_files()) - 1} referenced files."
                )
                self.refresh_config_files()
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load SUMO configuration file:\n{str(e)}"
                )
    
    def remove_sumocfg(self):
        """Remove the SUMO configuration file."""
        reply = QMessageBox.question(
            self,
            "Remove Configuration",
            "Are you sure you want to remove the SUMO configuration file?\n"
            "This will remove all associated files.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.remove_sumocfg()
            self.refresh_config_files()

