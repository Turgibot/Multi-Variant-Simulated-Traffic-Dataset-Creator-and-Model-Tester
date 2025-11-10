"""
Main application window with opening page and navigation.
"""

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QStackedWidget, QVBoxLayout,
                               QWidget)

from src.gui.dataset_generation_page import DatasetGenerationPage
from src.gui.project_dialog import NewProjectDialog
from src.gui.route_generation_page import RouteGenerationPage
from src.gui.simulation_page import SimulationPage
from src.utils.project_manager import ProjectManager


class WelcomePage(QWidget):
    """Opening page showing available projects and tool information."""
    
    # Signals
    dataset_generation_clicked = Signal(str)  # Emits project name
    model_testing_clicked = Signal(str)  # Emits project name
    new_project_created = Signal(str)  # Emits project name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.init_ui()
        self.refresh_projects()
    
    def init_ui(self):
        """Initialize the welcome page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title section
        title = QLabel("Traffic Simulation Tool")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Multi-Variant Dataset Creator and Model Tester")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        main_layout.addWidget(subtitle)
        
        # Tool description
        description = QLabel(
            "This tool provides an integrated environment for:\n"
            "• Creating diverse traffic simulation datasets with SUMO\n"
            "• Testing and evaluating traffic prediction models\n"
            "• Generating datasets in multiple formats (trajectory, sensor-based, GNN)\n"
            "• Analyzing model performance with statistical visualizations"
        )
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("""
            color: #555;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        """)
        main_layout.addWidget(description)
        
        # Projects section
        projects_label = QLabel("Projects")
        projects_label_font = QFont()
        projects_label_font.setPointSize(16)
        projects_label_font.setBold(True)
        projects_label.setFont(projects_label_font)
        main_layout.addWidget(projects_label)
        
        # Projects scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        self.projects_widget = QWidget()
        self.projects_layout = QVBoxLayout()
        self.projects_layout.setSpacing(10)
        self.projects_widget.setLayout(self.projects_layout)
        scroll.setWidget(self.projects_widget)
        main_layout.addWidget(scroll)
        
        # New project button
        new_project_btn = QPushButton("+ Create New Project")
        new_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        new_project_btn.clicked.connect(self.show_new_project_dialog)
        main_layout.addWidget(new_project_btn)
        
        # Version info
        try:
            from src import __version__
            version_text = f"Version {__version__}"
        except ImportError:
            version_text = "Version 0.1.0"
        version_label = QLabel(version_text)
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #999; font-size: 10px; margin-top: 10px;")
        main_layout.addWidget(version_label)
        
        self.setLayout(main_layout)
    
    def refresh_projects(self):
        """Refresh the projects list."""
        # Clear existing projects
        while self.projects_layout.count():
            item = self.projects_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get all projects
        projects = self.project_manager.get_all_projects()
        
        if not projects:
            # Show empty state
            empty_label = QLabel("No projects yet. Create a new project to get started!")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #999; padding: 20px;")
            self.projects_layout.addWidget(empty_label)
        else:
            # Add project cards
            for project in projects:
                card = self.create_project_card(project)
                self.projects_layout.addWidget(card)
        
        self.projects_layout.addStretch()
    
    def create_project_card(self, project: dict) -> QWidget:
        """Create a project card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
            QFrame:hover {
                background-color: #f0f0f0;
                border: 1px solid #4CAF50;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Delete button (left side)
        delete_btn = QPushButton("×")
        delete_btn.setToolTip("Delete project")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-size: 18px;
                font-weight: bold;
                min-width: 25px;
                max-width: 25px;
                min-height: 25px;
                max-height: 25px;
            }
            QPushButton:hover {
                background-color: #ffebee;
                color: #f44336;
            }
            QPushButton:pressed {
                background-color: #ffcdd2;
                color: #d32f2f;
            }
        """)
        delete_btn.clicked.connect(
            lambda: self.delete_project(project['name'])
        )
        layout.addWidget(delete_btn)
        
        # Project info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        name_label = QLabel(project['name'])
        name_font = QFont()
        name_font.setPointSize(14)
        name_font.setBold(True)
        name_label.setFont(name_font)
        info_layout.addWidget(name_label)
        
        path_label = QLabel(f"Path: {project['path']}")
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(path_label)
        
        if not project.get('exists', True):
            status_label = QLabel("⚠ Project folder not found")
            status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            info_layout.addWidget(status_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Buttons container
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        
        # Action buttons container
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(8)
        
        # Dataset Generation button
        dataset_btn = QPushButton("Dataset Generation")
        dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        dataset_btn.clicked.connect(
            lambda: self.dataset_generation_clicked.emit(project['name'])
        )
        action_buttons_layout.addWidget(dataset_btn)
        
        # Model Testing button
        model_btn = QPushButton("Model Testing")
        model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        model_btn.clicked.connect(
            lambda: self.model_testing_clicked.emit(project['name'])
        )
        action_buttons_layout.addWidget(model_btn)
        
        buttons_layout.addLayout(action_buttons_layout)
        
        layout.addLayout(buttons_layout)
        
        card.setLayout(layout)
        return card
    
    def show_new_project_dialog(self):
        """Show dialog to create a new project."""
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.Accepted:
            try:
                project_path = self.project_manager.create_project(
                    dialog.project_name,
                    dialog.project_description,
                    dialog.project_path
                )
                if project_path:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Project '{dialog.project_name}' created successfully!\n"
                        f"Location: {project_path}"
                    )
                    self.refresh_projects()
                    self.new_project_created.emit(dialog.project_name)
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Project '{dialog.project_name}' already exists!"
                    )
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def delete_project(self, project_name: str):
        """Delete a project."""
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Are you sure you want to delete project '{project_name}'?\n\n"
            "This will permanently delete the project folder and all its contents.\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                success = self.project_manager.delete_project(project_name)
                if success:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Project '{project_name}' deleted successfully."
                    )
                    self.refresh_projects()
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Project '{project_name}' not found."
                    )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to delete project:\n{str(e)}"
                )


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Traffic Simulation Tool - Multi-Variant Dataset Creator and Model Tester")
        self.setGeometry(100, 100, 1200, 800)
        self.showMaximized()
        
        # Central widget with stacked pages
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Welcome page
        self.welcome_page = WelcomePage()
        self.welcome_page.dataset_generation_clicked.connect(self.open_dataset_generation)
        self.welcome_page.model_testing_clicked.connect(self.open_model_testing)
        self.welcome_page.new_project_created.connect(self.on_new_project_created)
        self.central_widget.addWidget(self.welcome_page)
        
        # Dataset generation page (will be created when needed)
        self.dataset_page = None
        self.current_project_name = None
        self.current_project_path = None
        
        # Route generation page (will be created when needed)
        self.route_page = None
        
        # Simulation page (will be created when needed)
        self.simulation_page = None
        
        # Placeholder page for model testing
        self.model_page = self.create_placeholder_page("Model Testing", "Coming soon...")
        self.central_widget.addWidget(self.model_page)
        
        # Show welcome page
        self.central_widget.setCurrentWidget(self.welcome_page)
    
    def create_placeholder_page(self, title, message):
        """Create a placeholder page for features not yet implemented."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Back button
        back_btn = QPushButton("← Back to Home")
        back_btn.clicked.connect(self.show_welcome)
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
        layout.addWidget(back_btn)
        
        # Title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Message
        msg_label = QLabel(message)
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(msg_label)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def show_welcome(self):
        """Show the welcome page."""
        self.welcome_page.refresh_projects()
        self.central_widget.setCurrentWidget(self.welcome_page)
    
    def open_dataset_generation(self, project_name: str):
        """Open dataset generation for a project."""
        # Get project info
        project_info = self.welcome_page.project_manager.get_project_info(project_name)
        if not project_info:
            QMessageBox.warning(
                self,
                "Error",
                f"Project '{project_name}' not found."
            )
            return
        
        project_path = project_info['path']
        
        # Check if project folder exists
        if not Path(project_path).exists():
            QMessageBox.warning(
                self,
                "Error",
                f"Project folder does not exist: {project_path}"
            )
            return
        
        # Create or reuse dataset generation page
        if (self.dataset_page is None or 
            self.current_project_name != project_name):
            # Remove old page if exists
            if self.dataset_page is not None:
                self.central_widget.removeWidget(self.dataset_page)
                self.dataset_page.deleteLater()
            
            # Create new page
            self.dataset_page = DatasetGenerationPage(project_name, project_path)
            self.dataset_page.back_clicked.connect(self.show_welcome)
            self.dataset_page.run_simulation_clicked.connect(self.open_simulation)
            self.dataset_page.route_generation_clicked.connect(self.open_route_generation)
            self.central_widget.addWidget(self.dataset_page)
            self.current_project_name = project_name
            self.current_project_path = project_path
        
        # Show dataset generation page
        self.central_widget.setCurrentWidget(self.dataset_page)
    
    def open_route_generation(self):
        """Open route generation page for current project."""
        if not hasattr(self, 'current_project_name') or not self.current_project_name:
            QMessageBox.warning(
                self,
                "Error",
                "No project selected."
            )
            return
        
        project_name = self.current_project_name
        project_path = self.current_project_path
        
        # Create or reuse route generation page
        if (not hasattr(self, 'route_page') or 
            self.route_page is None or
            self.current_project_name != project_name):
            # Remove old page if exists
            if hasattr(self, 'route_page') and self.route_page is not None:
                self.central_widget.removeWidget(self.route_page)
                self.route_page.deleteLater()
            
            # Create new page
            self.route_page = RouteGenerationPage(project_name, project_path)
            self.route_page.back_clicked.connect(self.show_dataset_generation)
            self.central_widget.addWidget(self.route_page)
        
        # Show route generation page
        self.central_widget.setCurrentWidget(self.route_page)
    
    def show_dataset_generation(self):
        """Show dataset generation page."""
        if self.dataset_page is not None:
            self.central_widget.setCurrentWidget(self.dataset_page)
        else:
            # Reopen dataset generation if page doesn't exist
            if hasattr(self, 'current_project_name') and self.current_project_name:
                self.open_dataset_generation(self.current_project_name)
    
    def open_simulation(self, project_name: str, sumocfg_path: str, output_folder: str):
        """Open simulation page for a project."""
        # Get project info
        project_info = self.welcome_page.project_manager.get_project_info(project_name)
        if not project_info:
            QMessageBox.warning(
                self,
                "Error",
                f"Project '{project_name}' not found."
            )
            return
        
        project_path = project_info['path']
        
        # Check if project folder exists
        if not Path(project_path).exists():
            QMessageBox.warning(
                self,
                "Error",
                f"Project folder does not exist: {project_path}"
            )
            return
        
        # Create or reuse simulation page
        if (self.simulation_page is None or 
            self.current_project_name != project_name):
            # Remove old page if exists
            if self.simulation_page is not None:
                self.central_widget.removeWidget(self.simulation_page)
                self.simulation_page.deleteLater()
            
            # Create new page
            self.simulation_page = SimulationPage(
                project_name, 
                project_path, 
                sumocfg_path, 
                output_folder
            )
            self.simulation_page.back_clicked.connect(self.show_dataset_generation)
            self.central_widget.addWidget(self.simulation_page)
        
        # Show simulation page
        self.central_widget.setCurrentWidget(self.simulation_page)
    
    def show_dataset_generation(self):
        """Show the dataset generation page for the current project."""
        if self.dataset_page is not None:
            self.central_widget.setCurrentWidget(self.dataset_page)
        else:
            # If dataset page doesn't exist, go back to welcome
            self.show_welcome()
    
    def open_model_testing(self, project_name: str):
        """Open model testing for a project."""
        # TODO: Implement model testing page
        # For now, show a placeholder
        QMessageBox.information(
            self,
            "Model Testing",
            f"Opening Model Testing for project: {project_name}\n\n"
            "This feature will be implemented next."
        )
        # self.central_widget.setCurrentWidget(self.model_page)
    
    def on_new_project_created(self, project_name: str):
        """Handle new project creation."""
        # Optionally open the new project
        pass

