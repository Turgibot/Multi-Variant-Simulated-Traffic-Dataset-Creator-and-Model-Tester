"""
SUMO configuration file management for projects.
Handles loading and saving SUMO configuration file paths.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List


class SUMOConfigManager:
    """Manages SUMO configuration files for a project."""
    
    def __init__(self, project_path: str):
        """
        Initialize SUMO config manager for a project.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        self.config_file = self.project_path / 'sumo_config.json'
        self._ensure_config_file()
    
    def get_dataset_output_folder(self) -> Optional[str]:
        """
        Get the dataset output folder path.
        
        Returns:
            Path to dataset output folder or None if not set
        """
        config = self._load_config()
        return config.get('dataset_output_folder')
    
    def set_dataset_output_folder(self, folder_path: str):
        """
        Set the dataset output folder path.
        
        Args:
            folder_path: Path to the output folder
        """
        # Validate path exists and is a directory
        path = Path(folder_path)
        if not path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Store absolute path
        config = self._load_config()
        config['dataset_output_folder'] = str(path.absolute())
        self._save_config(config)
    
    def get_sumo_home(self) -> Optional[str]:
        """
        Get the SUMO_HOME path.
        
        Returns:
            Path to SUMO_HOME or None if not set
        """
        config = self._load_config()
        return config.get('sumo_home')
    
    def set_sumo_home(self, sumo_home_path: str):
        """
        Set the SUMO_HOME path.
        
        Args:
            sumo_home_path: Path to the SUMO installation directory
        """
        # Validate path exists and is a directory
        path = Path(sumo_home_path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {sumo_home_path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {sumo_home_path}")
        
        # Check if it looks like a SUMO installation (has bin directory)
        bin_dir = path / 'bin'
        if not bin_dir.exists():
            raise ValueError(f"Path does not appear to be a SUMO installation (missing bin directory): {sumo_home_path}")
        
        # Store absolute path
        config = self._load_config()
        config['sumo_home'] = str(path.absolute())
        self._save_config(config)
    
    def _ensure_config_file(self):
        """Ensure config file exists."""
        if not self.config_file.exists():
            self._save_config({})
    
    def _load_config(self) -> Dict:
        """Load SUMO configuration from JSON file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_config(self, config: Dict):
        """Save SUMO configuration to JSON file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_sumocfg_path(self) -> Optional[str]:
        """
        Get the main SUMO configuration file path.
        
        Returns:
            Path to .sumocfg file or None if not set
        """
        config = self._load_config()
        return config.get('sumocfg')
    
    def get_config_files(self) -> Dict[str, str]:
        """
        Get all SUMO configuration file paths (parsed from .sumocfg).
        
        Returns:
            Dictionary with file type as key and path as value
            Keys: 'network', 'routes', 'additional', 'sumocfg', etc.
        """
        config = self._load_config()
        files = {}
        
        # Add main sumocfg file
        if config.get('sumocfg'):
            files['sumocfg'] = config['sumocfg']
        
        # Add parsed files from sumocfg
        if 'parsed_files' in config:
            files.update(config['parsed_files'])
        
        return files
    
    def set_sumocfg(self, sumocfg_path: str):
        """
        Set the main SUMO configuration file and parse it.
        
        Args:
            sumocfg_path: Path to the .sumocfg file
        """
        # Validate file exists
        path = Path(sumocfg_path)
        if not path.exists():
            raise ValueError(f"File does not exist: {sumocfg_path}")
        
        if not path.suffix == '.sumocfg' and not path.name.endswith('.sumocfg'):
            raise ValueError(f"File must be a .sumocfg file: {sumocfg_path}")
        
        # Parse the sumocfg file to extract referenced files
        parsed_files = self._parse_sumocfg(path)
        
        # Store configuration
        config = self._load_config()
        config['sumocfg'] = str(path.absolute())
        config['parsed_files'] = parsed_files
        self._save_config(config)
    
    def _parse_sumocfg(self, sumocfg_path: Path) -> Dict[str, str]:
        """
        Parse a .sumocfg file to extract referenced files.
        
        Args:
            sumocfg_path: Path to the .sumocfg file
            
        Returns:
            Dictionary with file type as key and absolute path as value
        """
        parsed_files = {}
        sumocfg_dir = sumocfg_path.parent
        
        try:
            tree = ET.parse(sumocfg_path)
            root = tree.getroot()
            
            # Find input elements (SUMO config uses <input> tag)
            input_elem = root.find('input')
            
            if input_elem is not None:
                # Standard SUMO config format with <input> element
                # Network file
                net_file = input_elem.get('net-file')
                if net_file:
                    net_path = (sumocfg_dir / net_file).resolve()
                    # Store path even if it doesn't exist (for debugging)
                    parsed_files['network'] = str(net_path)
                
                # Route files
                route_files = input_elem.get('route-files')
                if route_files:
                    # SUMO can have multiple route files separated by comma
                    route_list = []
                    for route_file in route_files.split(','):
                        route_file = route_file.strip()
                        if route_file:
                            route_path = (sumocfg_dir / route_file).resolve()
                            route_list.append(str(route_path))
                    if route_list:
                        parsed_files['routes'] = route_list[0] if len(route_list) == 1 else route_list
                
                # Additional files
                additional_files = input_elem.get('additional-files')
                if additional_files:
                    add_list = []
                    for add_file in additional_files.split(','):
                        add_file = add_file.strip()
                        if add_file:
                            add_path = (sumocfg_dir / add_file).resolve()
                            add_list.append(str(add_path))
                    if add_list:
                        parsed_files['additional'] = add_list[0] if len(add_list) == 1 else add_list
                
                # Configuration file (if referenced)
                config_file = input_elem.get('configuration-file')
                if config_file:
                    config_path = (sumocfg_dir / config_file).resolve()
                    parsed_files['config'] = str(config_path)
            else:
                # Try alternative: attributes might be on root element
                # Network file
                net_file = root.get('net-file')
                if net_file:
                    net_path = (sumocfg_dir / net_file).resolve()
                    parsed_files['network'] = str(net_path)
                
                # Route files
                route_files = root.get('route-files')
                if route_files:
                    route_list = []
                    for route_file in route_files.split(','):
                        route_file = route_file.strip()
                        if route_file:
                            route_path = (sumocfg_dir / route_file).resolve()
                            route_list.append(str(route_path))
                    if route_list:
                        parsed_files['routes'] = route_list[0] if len(route_list) == 1 else route_list
                
                # Additional files
                additional_files = root.get('additional-files')
                if additional_files:
                    add_list = []
                    for add_file in additional_files.split(','):
                        add_file = add_file.strip()
                        if add_file:
                            add_path = (sumocfg_dir / add_file).resolve()
                            add_list.append(str(add_path))
                    if add_list:
                        parsed_files['additional'] = add_list[0] if len(add_list) == 1 else add_list
            
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse .sumocfg file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading .sumocfg file: {e}")
        
        return parsed_files
    
    def remove_sumocfg(self):
        """Remove the SUMO configuration file and all parsed files."""
        config = self._load_config()
        if 'sumocfg' in config:
            del config['sumocfg']
        if 'parsed_files' in config:
            del config['parsed_files']
        self._save_config(config)
    
    def has_sumocfg(self) -> bool:
        """Check if project has a SUMO configuration file."""
        return self.get_sumocfg_path() is not None
    
    def validate_files(self) -> Dict[str, bool]:
        """
        Validate that all configured files still exist.
        
        Returns:
            Dictionary with file type as key and exists status as value
        """
        files = self.get_config_files()
        validation = {}
        for file_type, file_path in files.items():
            if isinstance(file_path, list):
                # Multiple files
                validation[file_type] = all(Path(p).exists() for p in file_path)
            else:
                validation[file_type] = Path(file_path).exists()
        return validation
    
    def get_missing_files(self) -> List[str]:
        """Get list of file types that are missing or don't exist."""
        validation = self.validate_files()
        return [file_type for file_type, exists in validation.items() if not exists]

