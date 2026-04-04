"""Resolve optional debug logs under the repository `.cursor/` directory."""

from pathlib import Path

from src.utils.project_manager import _get_project_root


def cursor_debug_path(filename: str) -> Path:
    path = _get_project_root() / ".cursor" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
