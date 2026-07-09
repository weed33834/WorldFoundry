"""Module for base_models -> three_dimensions -> general_3d -> dust3r -> __init__.py functionality."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
DUST3R_PACKAGE_ROOT = SOURCE_ROOT / "dust3r"
CROCO_ROOT = SOURCE_ROOT / "croco"


def ensure_import_paths() -> tuple[str, str]:
    """Expose the upstream DUSt3R and CroCo packages from the canonical integration."""
    paths = (str(SOURCE_ROOT), str(CROCO_ROOT))
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        if path not in sys.path:
            sys.path.insert(0, path)
    return paths


IMPORT_PATHS = ensure_import_paths()


__all__ = [
    "CROCO_ROOT",
    "DUST3R_PACKAGE_ROOT",
    "IMPORT_PATHS",
    "SOURCE_ROOT",
    "ensure_import_paths",
]
