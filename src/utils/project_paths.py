"""
Path helpers: show project-relative paths in the UI while resolving to absolute for I/O.

Relative segments use POSIX slashes. Paths under the project root are stored as relative
to that root when possible; paths outside the project stay absolute (or use ~ for HOME).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

PathLike = Union[str, Path]


def resolve_path(text: PathLike, project_root: PathLike) -> Path:
    """Resolve a user or config path: absolute, ~/…, or relative to project_root."""
    root = Path(project_root).expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = Path(project_root).expanduser()

    raw = str(text).strip() if text is not None else ""
    if not raw:
        return Path()

    if raw.startswith("~/"):
        p = Path.home() / raw[2:].lstrip("/\\")
    else:
        p = Path(raw.replace("\\", "/")).expanduser()

    if p.is_absolute():
        try:
            return p.resolve()
        except OSError:
            return p
    try:
        return (root / p).resolve()
    except OSError:
        return root / p


def to_display_path(path: PathLike, project_root: PathLike) -> str:
    """Short string for labels and line edits (project-relative, else ~, else absolute)."""
    if path is None or str(path).strip() == "":
        return ""

    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(path).expanduser()

    proj = Path(project_root).expanduser()
    try:
        proj = proj.resolve()
    except OSError:
        proj = Path(project_root).expanduser()

    try:
        rel = p.relative_to(proj)
        out = rel.as_posix()
        return "." if out == "" else out
    except ValueError:
        pass

    home = Path.home()
    try:
        rel = p.relative_to(home.resolve())
        return "~/" + rel.as_posix()
    except ValueError:
        return str(p)


def compact_path(path: PathLike, project_root: PathLike) -> str:
    """Serialize for JSON/config: relative to project when under it, else absolute path string."""
    raw = str(path).strip() if path is not None else ""
    if not raw:
        return ""

    p = resolve_path(raw, project_root)
    proj = Path(project_root).expanduser()
    try:
        proj = proj.resolve()
    except OSError:
        proj = Path(project_root).expanduser()

    try:
        rel = p.relative_to(proj)
        out = rel.as_posix()
        return "." if out == "" else out
    except ValueError:
        return str(p)


def resolve_dataset_output_layout(project_path: PathLike) -> Dict[str, object]:
    """
    Resolve dataset output paths for simulation export.

    Uses output_dir from simulation.config.json when set; otherwise falls back to
    dataset_output_folder in sumo_config.json (default: <project>/datasets).
    """
    project = Path(project_path).expanduser()
    try:
        project = project.resolve()
    except OSError:
        project = Path(project_path).expanduser()

    snapshot_interval_sec = 30
    output_dir_raw = ""
    sim_config_path = project / "simulation.config.json"
    if sim_config_path.exists():
        try:
            with open(sim_config_path, "r", encoding="utf-8") as f:
                sim_config = json.load(f)
            if isinstance(sim_config, dict):
                output_dir_raw = read_output_dir_from_sim_config(sim_config, project)
                snapshot_interval_sec = int(sim_config.get("snapshot_interval_sec", 30))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if output_dir_raw:
        output_folder = resolve_path(output_dir_raw, project)
        snapshots_dir = output_folder / "snapshots"
    else:
        from src.utils.sumo_config_manager import SUMOConfigManager

        output_raw = SUMOConfigManager(str(project)).get_dataset_output_folder()
        if output_raw:
            output_folder = Path(output_raw)
        else:
            output_folder = project / "datasets"
        snapshots_dir = output_folder / "snapshots"

    try:
        output_folder = output_folder.resolve()
    except OSError:
        output_folder = Path(output_folder)
    try:
        snapshots_dir = snapshots_dir.resolve()
    except OSError:
        snapshots_dir = Path(snapshots_dir)

    return {
        "output_folder": str(output_folder),
        "snapshots_dir": str(snapshots_dir),
        "snapshot_interval_sec": max(1, int(snapshot_interval_sec)),
    }


def read_output_dir_from_sim_config(sim_config: dict, project: PathLike) -> str:
    """Read output_dir from simulation config, with legacy snapshot_dir migration."""
    raw = str(sim_config.get("output_dir") or "").strip()
    if raw:
        return raw

    legacy = str(sim_config.get("snapshot_dir") or "").strip()
    if not legacy:
        return ""

    legacy_path = resolve_path(legacy, project)
    if legacy_path.name == "snapshots":
        return str(legacy_path.parent)
    return legacy
