"""
SUMO installation detector and auto-configuration.
Helps find SUMO installations and set SUMO_HOME automatically.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def auto_detect_sumo_home() -> Optional[str]:
    """
    Auto-detect SUMO_HOME from various sources.
    
    Returns:
        Path to SUMO_HOME or None if not found
    """
    # 1. Check if already set in environment
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home and Path(sumo_home).exists():
        bin_dir = Path(sumo_home) / 'bin'
        if bin_dir.exists():
            return str(Path(sumo_home).absolute())

    # 2. Pip / uv: PyPI package "eclipse-sumo" installs the `sumo` module with SUMO_HOME set
    try:
        import sumo as sumo_dist  # provided by eclipse-sumo; not the TraCI "sumolib" helper
        pip_home = getattr(sumo_dist, "SUMO_HOME", None)
        if pip_home:
            home = Path(str(pip_home))
            if (home / "bin").exists():
                return str(home.resolve())
    except ImportError:
        pass
    
    # 3. Check installation directory (for bundled SUMO)
    if getattr(sys, 'frozen', False):
        # Running as executable
        app_dir = Path(sys.executable).parent
        sumo_dir = app_dir / 'sumo'
        if (sumo_dir / 'bin').exists():
            return str(sumo_dir.absolute())
    
    # 4. Try to find SUMO binary and work backwards
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'sumo'], capture_output=True, text=True)
        else:  # Linux/macOS
            result = subprocess.run(['which', 'sumo'], capture_output=True, text=True)
        
        if result.returncode == 0:
            sumo_bin = result.stdout.strip().split('\n')[0]
            sumo_bin_path = Path(sumo_bin)
            # SUMO binary is in bin/, so go up one level
            sumo_home = sumo_bin_path.parent.parent
            if (sumo_home / 'bin').exists() and (sumo_home / 'tools').exists():
                return str(sumo_home.absolute())
    except Exception:
        pass
    
    # 5. Check common installation paths
    common_paths = []
    
    if os.name == 'nt':  # Windows
        common_paths = [
            Path(os.environ.get('ProgramFiles', 'C:\\Program Files')) / 'SUMO',
            Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')) / 'SUMO',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'SUMO',
            Path(os.environ.get('APPDATA', '')) / 'SUMO',
        ]
    else:  # Linux/macOS
        common_paths = [
            Path('/usr/share/sumo'),
            Path('/opt/sumo'),
            Path('/usr/local/share/sumo'),
            Path.home() / 'sumo',
            Path.home() / '.sumo',
        ]
    
    for path in common_paths:
        if path.exists() and (path / 'bin').exists() and (path / 'tools').exists():
            return str(path.absolute())
    
    return None


def sumo_root_has_cli(home: Optional[str]) -> bool:
    """True if this SUMO root has the main sumo executable under bin/ (not only data/typemaps)."""
    if not home or not str(home).strip():
        return False
    root = Path(str(home).strip())
    if not root.is_dir():
        return False
    if os.name == "nt":
        return (root / "bin" / "sumo.exe").is_file()
    return (root / "bin" / "sumo").is_file()


def effective_sumo_home(ui_value: Optional[str] = None) -> str:
    """
    SUMO_HOME for subprocesses and data files: try config/UI path, then $SUMO_HOME,
    then auto_detect — but skip any candidate that has no bin/sumo (e.g. stale /usr/share/sumo).
    """
    seen = set()
    candidates = []

    if ui_value and str(ui_value).strip():
        candidates.append(str(ui_value).strip())
    env = os.environ.get("SUMO_HOME", "")
    if env.strip():
        candidates.append(env.strip())

    found = auto_detect_sumo_home()
    if found:
        candidates.append(found)

    try:
        import sumo as sumo_dist

        pip_home = getattr(sumo_dist, "SUMO_HOME", None)
        if pip_home:
            candidates.append(str(Path(str(pip_home)).resolve()))
    except ImportError:
        pass

    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if sumo_root_has_cli(c):
            return c

    return ""


def find_sumo_binary(sumo_home: Optional[str] = None) -> Optional[str]:
    """
    Find SUMO binary path.
    
    Args:
        sumo_home: Optional SUMO_HOME path. If None, tries to auto-detect.
        
    Returns:
        Path to SUMO binary or None if not found
    """
    if sumo_home is None:
        sumo_home = auto_detect_sumo_home()
    
    if sumo_home is None:
        return None
    
    sumo_home_path = Path(sumo_home)
    
    if os.name == 'nt':  # Windows
        sumo_binary = sumo_home_path / 'bin' / 'sumo.exe'
    else:  # Linux/macOS
        sumo_binary = sumo_home_path / 'bin' / 'sumo'
    
    if sumo_binary.exists():
        return str(sumo_binary.absolute())
    
    return None


def setup_sumo_environment(sumo_home: str):
    """
    Set up SUMO environment variables and Python path.
    
    Args:
        sumo_home: Path to SUMO_HOME
    """
    # Set SUMO_HOME environment variable
    os.environ['SUMO_HOME'] = sumo_home
    
    # Add SUMO tools to Python path
    tools_path = os.path.join(sumo_home, 'tools')
    if os.path.exists(tools_path) and tools_path not in sys.path:
        sys.path.insert(0, tools_path)

