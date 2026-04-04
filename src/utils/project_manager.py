"""
Project management utilities.
Handles project creation, loading, and registry management.
"""

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


def _resolve_registry_path(path_str: str) -> Path:
    """Resolve a path from the registry; relative paths are anchored to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _get_project_root() / p


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Try multiple strategies to find project root
    
    # Strategy 1: Look for directory with BOTH 'projects' AND 'src' (most reliable)
    current = Path.cwd()
    check_dir = current
    for _ in range(10):  # Check up to 10 levels
        if (check_dir / 'projects').exists() and (check_dir / 'src').exists():
            return check_dir
        if check_dir == check_dir.parent:  # Reached filesystem root
            break
        check_dir = check_dir.parent
    
    # Strategy 2: Use sys.path to find where 'src' module is located
    import sys
    for path_str in sys.path:
        try:
            path = Path(path_str).resolve()
            # If this path contains 'src', the parent is likely project root
            if path.name == 'src' and (path.parent / 'projects').exists():
                return path.parent
            # If this is the project root itself
            if (path / 'src').exists() and (path / 'projects').exists():
                return path
        except (ValueError, OSError):
            continue
    
    # Strategy 3: Fallback - look for just 'projects' directory
    check_dir = current
    for _ in range(10):
        if (check_dir / 'projects').exists():
            return check_dir
        if check_dir == check_dir.parent:
            break
        check_dir = check_dir.parent
    
    # Final fallback: use current directory
    return current


def ensure_porto_example_net_xml() -> None:
    """
    Extract examples/porto_conversion/config/porto.net.xml from the bundled zip if missing.
    The raw network is large; the repo ships porto.net.xml.zip (~20 MB) instead.
    """
    root = _get_project_root()
    config_dir = root / "examples" / "porto_conversion" / "config"
    net = config_dir / "porto.net.xml"
    if net.is_file():
        return
    zpath = config_dir / "porto.net.xml.zip"
    if not zpath.is_file():
        return
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            if "porto.net.xml" not in zf.namelist():
                return
            zf.extract("porto.net.xml", config_dir)
        print(f"DEBUG: extracted {net} from {zpath.name}")
    except (OSError, zipfile.BadZipFile, KeyError) as e:
        print(f"DEBUG: could not extract porto.net.xml: {e}")


def ensure_porto_example_train_csv() -> None:
    """
    Extract examples/porto_conversion/data/train.csv from bundled zip (single file or split parts).
    Large CSV is shipped as train.csv.zip split into ~95 MB parts for Git hosting limits.
    """
    root = _get_project_root()
    data_dir = root / "examples" / "porto_conversion" / "data"
    train = data_dir / "train.csv"
    if train.is_file():
        return

    single = data_dir / "train.csv.zip"
    parts = sorted(data_dir.glob("train.csv.zip.part*"))
    temp_zip: Optional[Path] = None
    zip_path: Optional[Path] = None
    try:
        if single.is_file():
            zip_path = single
        elif parts:
            temp_zip = data_dir / ".train_csv_bundle.zip"
            with open(temp_zip, "wb") as out:
                for p in parts:
                    with open(p, "rb") as inf:
                        shutil.copyfileobj(inf, out)
            zip_path = temp_zip
        else:
            return

        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if "train.csv" not in names:
                return
            zf.extract("train.csv", data_dir)
        print(f"DEBUG: extracted {train} from bundled train.csv.zip (parts)")
    except (OSError, zipfile.BadZipFile, KeyError) as e:
        print(f"DEBUG: could not extract train.csv: {e}")
    finally:
        if temp_zip is not None and temp_zip.is_file():
            try:
                temp_zip.unlink()
            except OSError:
                pass


def ensure_porto_example_bundled_assets() -> None:
    """Unpack shipped example artifacts (network, Porto CSVs) once if missing."""
    ensure_porto_example_net_xml()
    ensure_porto_example_train_csv()


def _print_sample_trajectory(project_root: Path) -> None:
    """Print a single trajectory from CSV in key-value format (key = CSV header)."""
    import csv
    for name in ("test.csv", "train.csv"):
        csv_path = project_root / "Porto" / "dataset" / name
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    row = next(reader, None)
                    if header and row:
                        print("DEBUG Sample trajectory (key=header):")
                        for k, v in zip(header, row):
                            key = str(k).strip('"')
                            val = str(v).strip('"')
                            if len(val) > 80:
                                val = val[:77] + "..."
                            print(f"  {key}: {val}")
                    break
            except Exception as e:
                print(f"DEBUG Sample trajectory: {e}")
            break


class ProjectManager:
    """Manages projects and their registry."""
    
    def __init__(self, projects_dir: str = "projects", registry_file: str = "projects_registry.json"):
        """
        Initialize project manager.
        
        Args:
            projects_dir: Directory where projects are stored
            registry_file: Name of the JSON registry file
        """
        # Resolve to absolute path based on project root
        if Path(projects_dir).is_absolute():
            self.projects_dir = Path(projects_dir)
        else:
            # Get project root and use it to resolve projects directory
            project_root = _get_project_root()
            self.projects_dir = project_root / projects_dir
        
        self.registry_file = self.projects_dir / registry_file
        self.projects_dir.mkdir(exist_ok=True, parents=True)

        ensure_porto_example_bundled_assets()
        self._ensure_registry()
        
        # Debug output
        print(f"DEBUG ProjectManager: project_root={_get_project_root()}")
        print(f"DEBUG ProjectManager: projects_dir={self.projects_dir}")
        print(f"DEBUG ProjectManager: registry_file={self.registry_file}")
        print(f"DEBUG ProjectManager: registry_file exists={self.registry_file.exists()}")
        
        if self.registry_file.exists():
            registry = self._load_registry()
            print(f"DEBUG ProjectManager: registry has {len(registry)} projects: {list(registry.keys())}")

        # Print a single trajectory in key-value format (key = CSV header)
        _print_sample_trajectory(_get_project_root())
    
    def _try_seed_registry_from_examples(self) -> None:
        """If bundled example registry exists, copy it once (when projects registry is missing)."""
        example = _get_project_root() / "examples" / "projects_registry.json"
        if not example.is_file():
            return
        try:
            with open(example, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except (OSError, json.JSONDecodeError):
            pass
    
    def _ensure_registry(self):
        """Ensure registry file exists; on first run seed from examples/projects_registry.json if present."""
        if not self.registry_file.exists():
            self._try_seed_registry_from_examples()
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
    
    def get_all_projects(self, project_type: Optional[str] = None) -> List[Dict]:
        """
        Get all registered projects, optionally filtered by type.
        
        Args:
            project_type: Filter by project type ('simulation', 'porto', or None for all)
        
        Returns:
            List of project dictionaries with 'name', 'path', 'type' keys
        """
        registry = self._load_registry()
        projects = []
        for name, data in registry.items():
            # Handle both old format (string path) and new format (dict with path and type)
            if isinstance(data, str):
                # Old format: just path string, assume simulation type
                path = data
                ptype = 'simulation'
            else:
                # New format: dict with path and type
                path = data.get('path', '')
                ptype = data.get('type', 'simulation')
            
            # Filter by type if specified
            if project_type is not None and ptype != project_type:
                continue
            
            project_path = _resolve_registry_path(path)
            projects.append({
                'name': name,
                'path': str(project_path),
                'type': ptype,
                'exists': project_path.exists()
            })
        return projects
    
    def create_project(self, name: str, description: str = "", base_path: Optional[str] = None, 
                       project_type: str = "simulation") -> Optional[str]:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            base_path: Base directory where project will be created. If None, uses default projects_dir.
            project_type: Type of project ('simulation' or 'porto')
            
        Returns:
            Project path if successful, None if project already exists
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        
        # Validate project type
        if project_type not in ('simulation', 'porto'):
            raise ValueError(f"Invalid project type: {project_type}. Must be 'simulation' or 'porto'.")
        
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
            'type': project_type,
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
        
        # Register project with type
        registry[name] = {
            'path': str(project_path),
            'type': project_type
        }
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
        
        data = registry[name]
        # Handle both old format (string path) and new format (dict with path and type)
        if isinstance(data, str):
            project_path = _resolve_registry_path(data)
        else:
            project_path = _resolve_registry_path(data.get('path', ''))
        
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
        
        data = registry[name]
        # Handle both old format (string path) and new format (dict with path and type)
        if isinstance(data, str):
            project_path = _resolve_registry_path(data)
            project_type = 'simulation'
        else:
            project_path = _resolve_registry_path(data.get('path', ''))
            project_type = data.get('type', 'simulation')
        
        info_file = project_path / 'project_info.json'
        
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    # Ensure type is included
                    if 'type' not in info:
                        info['type'] = project_type
                    if 'path' in info:
                        info['path'] = str(_resolve_registry_path(info['path']))
                    return info
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        return {
            'name': name,
            'path': str(project_path),
            'type': project_type,
            'description': ''
        }

