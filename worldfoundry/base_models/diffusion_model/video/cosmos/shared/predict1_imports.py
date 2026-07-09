"""Module for base_models -> diffusion_model -> video -> cosmos -> shared -> predict1_imports.py functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def ensure_predict1_import_path(package_file: str, module: ModuleType | None, alias_top_level: bool = False) -> str:
    """Ensure predict1 import path.

    Args:
        package_file: The package file.
        module: The module.
        alias_top_level: The alias top level.

    Returns:
        The return value.
    """
    source_root = Path(package_file).resolve().parent.parent
    path = str(source_root)
    if not sys.path or sys.path[0] != path:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    if alias_top_level and module is not None:
        sys.modules["cosmos_predict1"] = module
    return path
