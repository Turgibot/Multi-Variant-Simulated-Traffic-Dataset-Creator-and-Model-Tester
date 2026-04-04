"""
Path helpers: show project-relative paths in the UI while resolving to absolute for I/O.

Relative segments use POSIX slashes. Paths under the project root are stored as relative
to that root when possible; paths outside the project stay absolute (or use ~ for HOME).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

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
