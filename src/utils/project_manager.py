"""
Project management utilities.
Handles project creation, loading, and registry management.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional


class ProjectManager:
    """Manages projects and their registry."""
    
    def __init__(self, projects_dir: str = "projects", registry_file: str = "projects_registry.json"):
        """
        Initialize project manager.
        
        Args:
            projects_dir: Directory where projects are stored
            registry_file: Name of the JSON registry file
        """
        self.projects_dir = Path(projects_dir)
        self.registry_file = self.projects_dir / registry_file
        self.projects_dir.mkdir(exist_ok=True)
        self._ensure_registry()
    
    def _ensure_registry(self):
        """Ensure registry file exists."""
        if not self.registry_file.exists():
            self._save_registry({})
    
    def _load_registry(self) -> Dict:
        """Load projects registry from JSON file."""
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_registry(self, registry: Dict):
        """Save projects registry to JSON file."""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def get_all_projects(self) -> List[Dict]:
        """
        Get all registered projects.
        
        Returns:
            List of project dictionaries with 'name' and 'path' keys
        """
        registry = self._load_registry()
        projects = []
        for name, path in registry.items():
            project_path = Path(path)
            if project_path.exists():
                projects.append({
                    'name': name,
                    'path': str(project_path),
                    'exists': True
                })
            else:
                projects.append({
                    'name': name,
                    'path': str(project_path),
                    'exists': False
                })
        return projects
    
    def create_project(self, name: str, description: str = "", base_path: Optional[str] = None) -> Optional[str]:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            base_path: Base directory where project will be created. If None, uses default projects_dir.
            
        Returns:
            Project path if successful, None if project already exists
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        
        # Sanitize name for folder
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            raise ValueError("Project name must contain at least one alphanumeric character")
        
        # Check if project already exists
        registry = self._load_registry()
        if name in registry:
            return None
        
        # Determine project path
        if base_path:
            base_path_obj = Path(base_path)
            if not base_path_obj.exists() or not base_path_obj.is_dir():
                raise ValueError(f"Base path does not exist or is not a directory: {base_path}")
            project_path = base_path_obj / safe_name
        else:
            project_path = self.projects_dir / safe_name
        
        # Check if project folder already exists
        if project_path.exists():
            raise ValueError(f"Project folder already exists: {project_path}")
        
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create project info file
        project_info = {
            'name': name,
            'description': description,
            'created': str(Path().cwd()),
            'path': str(project_path)
        }
        
        info_file = project_path / 'project_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(project_info, f, indent=2, ensure_ascii=False)
        
        # Create subdirectories
        (project_path / 'config').mkdir(exist_ok=True)
        (project_path / 'data').mkdir(exist_ok=True)
        (project_path / 'results').mkdir(exist_ok=True)
        (project_path / 'models').mkdir(exist_ok=True)
        (project_path / 'datasets').mkdir(exist_ok=True)
        
        # Register project
        registry[name] = str(project_path)
        self._save_registry(registry)
        
        return str(project_path)
    
    def delete_project(self, name: str) -> bool:
        """
        Delete a project.
        
        Args:
            name: Project name
            
        Returns:
            True if successful, False if project not found
        """
        registry = self._load_registry()
        if name not in registry:
            return False
        
        project_path = Path(registry[name])
        
        # Remove from registry
        del registry[name]
        self._save_registry(registry)
        
        # Delete project directory
        if project_path.exists():
            import shutil
            shutil.rmtree(project_path)
        
        return True
    
    def get_project_info(self, name: str) -> Optional[Dict]:
        """
        Get project information.
        
        Args:
            name: Project name
            
        Returns:
            Project info dictionary or None if not found
        """
        registry = self._load_registry()
        if name not in registry:
            return None
        
        project_path = Path(registry[name])
        info_file = project_path / 'project_info.json'
        
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        return {
            'name': name,
            'path': str(project_path),
            'description': ''
        }

