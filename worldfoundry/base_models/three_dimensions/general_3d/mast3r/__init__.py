"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> __init__.py functionality."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
GENERAL_3D_ROOT = SOURCE_ROOT.parent
DUST3R_ROOT = GENERAL_3D_ROOT / "dust3r"


def ensure_import_paths() -> tuple[str, str]:
    """Expose upstream MASt3R and DUSt3R packages without nesting them in model runtimes."""
    paths = (str(SOURCE_ROOT), str(DUST3R_ROOT))
    for path in paths:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return paths


def reexport_dust3r():
    """Expose the canonical DUSt3R integration as ``worldfoundry...mast3r.dust3r``."""
    ensure_import_paths()
    module = importlib.import_module(
        "worldfoundry.base_models.three_dimensions.general_3d.dust3r"
    )
    sys.modules[f"{__name__}.dust3r"] = module
    return module


dust3r = reexport_dust3r()


__all__ = [
    "DUST3R_ROOT",
    "GENERAL_3D_ROOT",
    "SOURCE_ROOT",
    "dust3r",
    "ensure_import_paths",
    "reexport_dust3r",
]
