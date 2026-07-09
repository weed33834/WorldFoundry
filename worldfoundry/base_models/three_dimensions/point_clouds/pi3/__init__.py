"""Module for base_models -> three_dimensions -> point_clouds -> pi3 -> __init__.py functionality."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent


def ensure_import_paths() -> tuple[Path, ...]:
    """Expose the shared Pi3 source as the top-level ``pi3`` package."""

    source_root = str(SOURCE_ROOT)
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    return (SOURCE_ROOT,)


__all__ = ["SOURCE_ROOT", "ensure_import_paths"]
