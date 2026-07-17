"""Resolve upstream ViPE config class paths inside the WorldFoundry namespace."""

from __future__ import annotations

import importlib
from types import ModuleType

_IN_TREE_PACKAGE = "worldfoundry.base_models.three_dimensions.general_3d.vipe"


def import_config_module(module_path: str) -> ModuleType:
    """Import an official ``vipe.*`` path without installing a second package."""
    if module_path == "vipe":
        module_path = _IN_TREE_PACKAGE
    elif module_path.startswith("vipe."):
        module_path = f"{_IN_TREE_PACKAGE}{module_path[len('vipe') :]}"
    return importlib.import_module(module_path)


__all__ = ["import_config_module"]
