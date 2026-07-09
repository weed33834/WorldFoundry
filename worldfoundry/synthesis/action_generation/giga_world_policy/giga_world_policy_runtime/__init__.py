from __future__ import annotations

import importlib
import sys


def install_aliases() -> None:
    """Register upstream package names for the vendored GigaWorld runtime.

    Args:
        None.
    """
    package_root = __name__
    if "world_action_model" not in sys.modules:
        sys.modules["world_action_model"] = importlib.import_module(f"{package_root}.world_action_model")


__all__ = ["install_aliases"]
