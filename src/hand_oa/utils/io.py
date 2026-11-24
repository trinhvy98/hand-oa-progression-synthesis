"""Filesystem, project-root discovery, and sys.path bootstrap helpers."""

import json
import sys
from pathlib import Path
from typing import Any


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward from `start` to locate the repository root directory."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml or configs/")


def bootstrap_project_paths(start: str | Path | None = None) -> Path:
    """Ensure `<root>/src` and `<root>` are on `sys.path` and return the project root."""
    root = find_project_root(start)
    for path_entry in (str(root), str(root / "src")):
        if path_entry not in sys.path:
            sys.path.insert(0, path_entry)
    return root


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a relative path against the repository root."""
    target = Path(path)
    if target.is_absolute():
        return target
    root = Path(project_root) if project_root is not None else find_project_root()
    return (root / target).resolve()


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    """Write a dictionary to a formatted JSON file."""
    target = Path(output_path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return target
