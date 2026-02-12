"""
Main application window with opening page and navigation.
"""

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QStackedWidget, QVBoxLayout,
                               QWidget)

from src.gui.dataset_conversion_page import DatasetConversionPage
from src.gui.dataset_generation_page import DatasetGenerationPage
from src.gui.debug_trajectory_page import DebugTrajectoryPage
from src.gui.project_dialog import NewProjectDialog
from src.gui.route_generation_page import RouteGenerationPage
from src.gui.simulation_page import SimulationPage
from src.utils.project_manager import ProjectManager


class WelcomePage(QWidget):
    """Opening page showing available projects and tool information."""
    
    # Signals
    dataset_generation_clicked = Signal(str)  # Emits project name
    model_testing_clicked = Signal(str)  # Emits project name
    new_project_created = Signal(str, str)  # Emits project name and type
    porto_conversion_clicked = Signal(str)  # Emits project name for Porto conversion
    debug_trajectory_clicked = Signal(str)  # Emits project name for debug trajectory page
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.init_ui()
        self.refresh_projects()
    
    def init_ui(self):
        """Initialize the welcome page UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(30, 20, 30, 20)
        
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
        
        # Two-column layout for project sections
        projects_columns = QHBoxLayout()
        projects_columns.setSpacing(20)
        
        # ========== LEFT COLUMN: Simulation Projects ==========
        sim_section = QVBoxLayout()
        sim_section.setSpacing(10)
        
        # Simulation projects header
        sim_header = QHBoxLayout()
        sim_label = QLabel("🚗 Simulation Projects")
        sim_label_font = QFont()
        sim_label_font.setPointSize(14)
        sim_label_font.setBold(True)
        sim_label.setFont(sim_label_font)
        sim_label.setStyleSheet("color: #4CAF50;")
        sim_header.addWidget(sim_label)
        sim_header.addStretch()
        
        # Create simulation project button
        new_sim_btn = QPushButton("+ Create")
        new_sim_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        new_sim_btn.clicked.connect(self.show_new_simulation_project_dialog)
        sim_header.addWidget(new_sim_btn)
        sim_section.addLayout(sim_header)
        
        # Simulation projects scroll area
        sim_scroll = QScrollArea()
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        self.sim_projects_widget = QWidget()
        self.sim_projects_layout = QVBoxLayout()
        self.sim_projects_layout.setSpacing(8)
        self.sim_projects_layout.setContentsMargins(5, 5, 5, 5)
        self.sim_projects_widget.setLayout(self.sim_projects_layout)
        sim_scroll.setWidget(self.sim_projects_widget)
        sim_section.addWidget(sim_scroll)
        
        projects_columns.addLayout(sim_section)
        
        # ========== RIGHT COLUMN: Porto Conversion Projects ==========
        porto_section = QVBoxLayout()
        porto_section.setSpacing(10)
        
        # Porto projects header
        porto_header = QHBoxLayout()
        porto_label = QLabel("🚕 Porto Conversion Projects")
        porto_label_font = QFont()
        porto_label_font.setPointSize(14)
        porto_label_font.setBold(True)
        porto_label.setFont(porto_label_font)
        porto_label.setStyleSheet("color: #FF9800;")
        porto_header.addWidget(porto_label)
        porto_header.addStretch()
        
        # Create Porto project button
        new_porto_btn = QPushButton("+ Create")
        new_porto_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        new_porto_btn.clicked.connect(self.show_new_porto_project_dialog)
        porto_header.addWidget(new_porto_btn)
        porto_section.addLayout(porto_header)
        
        # Porto projects scroll area
        porto_scroll = QScrollArea()
        porto_scroll.setWidgetResizable(True)
        porto_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #FF9800;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        self.porto_projects_widget = QWidget()
        self.porto_projects_layout = QVBoxLayout()
        self.porto_projects_layout.setSpacing(8)
        self.porto_projects_layout.setContentsMargins(5, 5, 5, 5)
        self.porto_projects_widget.setLayout(self.porto_projects_layout)
        porto_scroll.setWidget(self.porto_projects_widget)
        porto_section.addWidget(porto_scroll)
        
        projects_columns.addLayout(porto_section)
        
        main_layout.addLayout(projects_columns)
        
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
        """Refresh both project lists."""
        self.refresh_simulation_projects()
        self.refresh_porto_projects()
    
    def refresh_simulation_projects(self):
        """Refresh the simulation projects list."""
        try:
            if not hasattr(self, 'sim_projects_layout'):
                return
            
            # Clear existing projects
            while self.sim_projects_layout.count():
                item = self.sim_projects_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Get simulation projects
            projects = self.project_manager.get_all_projects(project_type='simulation')
            
            if not projects:
                empty_label = QLabel("No simulation projects yet.\nClick '+ Create' to get started!")
                empty_label.setAlignment(Qt.AlignCenter)
                empty_label.setStyleSheet("color: #999; padding: 20px;")
                self.sim_projects_layout.addWidget(empty_label)
            else:
                for project in projects:
                    try:
                        card = self.create_simulation_project_card(project)
                        self.sim_projects_layout.addWidget(card)
                    except Exception as e:
                        print(f"ERROR: Failed to create sim card for {project.get('name', 'unknown')}: {e}")
            
            self.sim_projects_layout.addStretch()
        except Exception as e:
            print(f"ERROR in refresh_simulation_projects: {e}")
    
    def refresh_porto_projects(self):
        """Refresh the Porto projects list."""
        try:
            if not hasattr(self, 'porto_projects_layout'):
                return
            
            # Clear existing projects
            while self.porto_projects_layout.count():
                item = self.porto_projects_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Get Porto projects
            projects = self.project_manager.get_all_projects(project_type='porto')
            
            if not projects:
                empty_label = QLabel("No Porto conversion projects yet.\nClick '+ Create' to get started!")
                empty_label.setAlignment(Qt.AlignCenter)
                empty_label.setStyleSheet("color: #999; padding: 20px;")
                self.porto_projects_layout.addWidget(empty_label)
            else:
                for project in projects:
                    try:
                        card = self.create_porto_project_card(project)
                        self.porto_projects_layout.addWidget(card)
                    except Exception as e:
                        print(f"ERROR: Failed to create porto card for {project.get('name', 'unknown')}: {e}")
            
            self.porto_projects_layout.addStretch()
        except Exception as e:
            print(f"ERROR in refresh_porto_projects: {e}")
    
    def create_simulation_project_card(self, project: dict) -> QWidget:
        """Create a simulation project card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
            QFrame:hover {
                background-color: #E8F5E9;
                border: 1px solid #4CAF50;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Delete button (left side)
        delete_btn = QPushButton("×")
        delete_btn.setToolTip("Delete project")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999;
                border: none;
                padding: 3px;
                border-radius: 3px;
                font-size: 16px;
                font-weight: bold;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: #ffebee;
                color: #f44336;
            }
        """)
        delete_btn.clicked.connect(
            lambda: self.delete_project(project['name'], 'simulation')
        )
        layout.addWidget(delete_btn)
        
        # Project info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        name_label = QLabel(project['name'])
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        info_layout.addWidget(name_label)
        
        if not project.get('exists', True):
            status_label = QLabel("⚠ Folder not found")
            status_label.setStyleSheet("color: #f44336; font-size: 9px;")
            info_layout.addWidget(status_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(5)
        
        # Dataset Generation button
        dataset_btn = QPushButton("Dataset Generation")
        dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        dataset_btn.clicked.connect(
            lambda: self.dataset_generation_clicked.emit(project['name'])
        )
        action_layout.addWidget(dataset_btn)
        
        # Model Testing button
        model_btn = QPushButton("Model Testing")
        model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        model_btn.clicked.connect(
            lambda: self.model_testing_clicked.emit(project['name'])
        )
        action_layout.addWidget(model_btn)
        
        layout.addLayout(action_layout)
        
        card.setLayout(layout)
        return card
    
    def create_porto_project_card(self, project: dict) -> QWidget:
        """Create a Porto conversion project card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
            QFrame:hover {
                background-color: #FFF3E0;
                border: 1px solid #FF9800;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Delete button (left side)
        delete_btn = QPushButton("×")
        delete_btn.setToolTip("Delete project")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999;
                border: none;
                padding: 3px;
                border-radius: 3px;
                font-size: 16px;
                font-weight: bold;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: #ffebee;
                color: #f44336;
            }
        """)
        delete_btn.clicked.connect(
            lambda: self.delete_project(project['name'], 'porto')
        )
        layout.addWidget(delete_btn)
        
        # Project info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        name_label = QLabel(project['name'])
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        info_layout.addWidget(name_label)
        
        if not project.get('exists', True):
            status_label = QLabel("⚠ Folder not found")
            status_label.setStyleSheet("color: #f44336; font-size: 9px;")
            info_layout.addWidget(status_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(5)
        
        # Dataset Conversion button
        convert_btn = QPushButton("Dataset Conversion")
        convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        convert_btn.clicked.connect(
            lambda: self.porto_conversion_clicked.emit(project['name'])
        )
        action_layout.addWidget(convert_btn)
        
        # Model Testing button
        model_btn = QPushButton("Model Testing")
        model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        model_btn.clicked.connect(
            lambda: self.model_testing_clicked.emit(project['name'])
        )
        action_layout.addWidget(model_btn)
        
        layout.addLayout(action_layout)
        
        card.setLayout(layout)
        return card
    
    def show_new_simulation_project_dialog(self):
        """Show dialog to create a new simulation project."""
        dialog = NewProjectDialog(self, project_type="simulation")
        if dialog.exec() == QDialog.Accepted:
            try:
                project_path = self.project_manager.create_project(
                    dialog.project_name,
                    dialog.project_description,
                    dialog.project_path,
                    project_type="simulation"
                )
                if project_path:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Simulation project '{dialog.project_name}' created successfully!\n"
                        f"Location: {project_path}"
                    )
                    self.refresh_projects()
                    self.new_project_created.emit(dialog.project_name, "simulation")
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Project '{dialog.project_name}' already exists!"
                    )
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def show_new_porto_project_dialog(self):
        """Show dialog to create a new Porto conversion project."""
        dialog = NewProjectDialog(self, project_type="porto")
        if dialog.exec() == QDialog.Accepted:
            try:
                project_path = self.project_manager.create_project(
                    dialog.project_name,
                    dialog.project_description,
                    dialog.project_path,
                    project_type="porto"
                )
                if project_path:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Porto conversion project '{dialog.project_name}' created successfully!\n"
                        f"Location: {project_path}"
                    )
                    self.refresh_projects()
                    self.new_project_created.emit(dialog.project_name, "porto")
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Project '{dialog.project_name}' already exists!"
                    )
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def delete_project(self, project_name: str, project_type: str = "simulation"):
        """Delete a project."""
        type_label = "simulation" if project_type == "simulation" else "Porto conversion"
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Are you sure you want to delete {type_label} project '{project_name}'?\n\n"
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
        
        # Set maximum size to screen size to prevent overflow
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.availableGeometry()
            self.setMaximumSize(screen_size.width(), screen_size.height())
        
        self.showMaximized()
        
        # Central widget with stacked pages
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Welcome page
        self.welcome_page = WelcomePage()
        self.welcome_page.dataset_generation_clicked.connect(self.open_dataset_generation)
        self.welcome_page.model_testing_clicked.connect(self.open_model_testing)
        self.welcome_page.new_project_created.connect(self.on_new_project_created)
        self.welcome_page.porto_conversion_clicked.connect(self.open_porto_conversion)
        self.welcome_page.debug_trajectory_clicked.connect(self.open_debug_trajectory)
        self.central_widget.addWidget(self.welcome_page)
        
        # Porto conversion page (will be created when needed)
        self.porto_page = None
        
        # Debug trajectory page (will be created when needed)
        self.debug_trajectory_page = None
        
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
        
        # Always check if we need to recreate the dataset page for the current project
        # Check if dataset_page exists and belongs to a different project
        dataset_page_project = None
        if self.dataset_page is not None:
            # Get the project name from the dataset page
            dataset_page_project = getattr(self.dataset_page, 'project_name', None)
        
        # Create or reuse dataset generation page
        if (self.dataset_page is None or 
            dataset_page_project != project_name):
            # Remove old page if exists
            if self.dataset_page is not None:
                self.central_widget.removeWidget(self.dataset_page)
                self.dataset_page.deleteLater()
                self.dataset_page = None
            
            # Create new page
            self.dataset_page = DatasetGenerationPage(project_name, project_path)
            self.dataset_page.back_clicked.connect(self.show_welcome)
            self.dataset_page.run_simulation_clicked.connect(self.open_simulation)
            self.dataset_page.route_generation_clicked.connect(self.open_route_generation)
            self.central_widget.addWidget(self.dataset_page)
            self.current_project_name = project_name
            self.current_project_path = project_path
        
        # Always show dataset generation page (not route generation)
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
        
        # Always check if we need to recreate the route page for the current project
        # Check if route_page exists and belongs to a different project
        route_page_project = None
        if hasattr(self, 'route_page') and self.route_page is not None:
            # Get the project name from the route page
            route_page_project = getattr(self.route_page, 'project_name', None)
        
        # Create or reuse route generation page
        if (not hasattr(self, 'route_page') or 
            self.route_page is None or
            route_page_project != project_name):
            # Remove old page if exists
            if hasattr(self, 'route_page') and self.route_page is not None:
                self.central_widget.removeWidget(self.route_page)
                self.route_page.deleteLater()
                self.route_page = None
            
            # Create new page for the current project
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
    
    def on_new_project_created(self, project_name: str, project_type: str):
        """Handle new project creation."""
        # Optionally auto-open the new project
        if project_type == "simulation":
            self.open_dataset_generation(project_name)
        elif project_type == "porto":
            self.open_porto_conversion(project_name)
    
    def open_porto_conversion(self, project_name: str):
        """Open Porto taxi dataset conversion page for a project."""
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
        
        # Check if we need to recreate the porto page for the current project
        porto_page_project = None
        if self.porto_page is not None:
            porto_page_project = getattr(self.porto_page, 'project_name', None)
        
        # Create or reuse Porto conversion page
        if (self.porto_page is None or porto_page_project != project_name):
            # Remove old page if exists
            if self.porto_page is not None:
                self.central_widget.removeWidget(self.porto_page)
                self.porto_page.deleteLater()
                self.porto_page = None
            
            # Create new page
            self.porto_page = DatasetConversionPage(project_name, project_path)
            self.porto_page.back_clicked.connect(self.show_welcome)
            self.central_widget.addWidget(self.porto_page)
        
        # Show Porto conversion page
        self.central_widget.setCurrentWidget(self.porto_page)
    
    def open_debug_trajectory(self, project_name: str):
        """Open debug trajectory page for a project."""
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
        
        # Check if we need to recreate the debug page for the current project
        debug_page_project = None
        if self.debug_trajectory_page is not None:
            debug_page_project = getattr(self.debug_trajectory_page, 'project_name', None)
        
        # Create or reuse debug trajectory page
        if (self.debug_trajectory_page is None or debug_page_project != project_name):
            # Remove old page if exists
            if self.debug_trajectory_page is not None:
                self.central_widget.removeWidget(self.debug_trajectory_page)
                self.debug_trajectory_page.deleteLater()
                self.debug_trajectory_page = None
            
            # Create new debug page
            self.debug_trajectory_page = DebugTrajectoryPage(project_name, project_path)
            self.debug_trajectory_page.back_clicked.connect(self.show_welcome)
            self.central_widget.addWidget(self.debug_trajectory_page)
        
        # Show debug trajectory page
        self.central_widget.setCurrentWidget(self.debug_trajectory_page)

