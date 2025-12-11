"""
Dialog for creating a new project.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt
from pathlib import Path


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""
    
    def __init__(self, parent=None, project_type: str = "simulation"):
        super().__init__(parent)
        self.project_type = project_type
        
        # Set title based on project type
        if project_type == "porto":
            self.setWindowTitle("Create Porto Conversion Project")
            self.title_text = "Create Porto Conversion Project"
            self.description_hint = "Project for converting Porto taxi GPS data to traffic simulation..."
        else:
            self.setWindowTitle("Create Simulation Project")
            self.title_text = "Create Simulation Project"
            self.description_hint = "Project for SUMO-based traffic simulation..."
        
        self.setModal(True)
        self.setMinimumWidth(500)
        self.project_name = None
        self.project_description = None
        self.project_path = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel(self.title_text)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # Project type indicator
        if self.project_type == "porto":
            type_label = QLabel("🚕 Porto Taxi Dataset Conversion")
            type_label.setStyleSheet("color: #FF9800; font-size: 12px; font-weight: bold; padding: 5px; background-color: #FFF3E0; border-radius: 3px;")
        else:
            type_label = QLabel("🚗 SUMO Traffic Simulation")
            type_label.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold; padding: 5px; background-color: #E8F5E9; border-radius: 3px;")
        layout.addWidget(type_label)
        
        # Project name
        name_label = QLabel("Project Name:")
        layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter project name...")
        layout.addWidget(self.name_input)
        
        # Project description
        desc_label = QLabel("Description (optional):")
        layout.addWidget(desc_label)
        
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText(self.description_hint)
        self.desc_input.setMaximumHeight(100)
        layout.addWidget(self.desc_input)
        
        # Project path
        path_label = QLabel("Project Location:")
        layout.addWidget(path_label)
        
        path_layout = QHBoxLayout()
        path_layout.setSpacing(10)
        
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select folder where project will be created...")
        self.path_input.setReadOnly(True)
        path_layout.addWidget(self.path_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet("""
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
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)
        
        layout.addLayout(path_layout)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create")
        create_btn.setStyleSheet("""
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
        create_btn.clicked.connect(self.create_project)
        buttons_layout.addWidget(create_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def browse_path(self):
        """Open file dialog to select project location."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Project Location",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if path:
            self.path_input.setText(path)
            self.project_path = path
    
    def create_project(self):
        """Validate and create project."""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Project name cannot be empty."
            )
            return
        
        path = self.path_input.text().strip()
        if not path:
            QMessageBox.warning(
                self,
                "Invalid Path",
                "Please select a location for the project."
            )
            return
        
        # Validate path exists
        path_obj = Path(path)
        if not path_obj.exists():
            QMessageBox.warning(
                self,
                "Invalid Path",
                "Selected path does not exist."
            )
            return
        
        if not path_obj.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Path",
                "Selected path is not a directory."
            )
            return
        
        self.project_name = name
        self.project_description = self.desc_input.toPlainText().strip()
        self.project_path = path
        self.accept()

