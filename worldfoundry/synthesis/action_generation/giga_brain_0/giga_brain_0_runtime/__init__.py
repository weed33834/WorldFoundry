from __future__ import annotations

import importlib
import sys


def install_aliases() -> None:
    """Register upstream package names for the vendored GigaBrain runtime.

    Args:
        None.
    """
    package_root = __name__
    if "giga_models" not in sys.modules:
        sys.modules["giga_models"] = importlib.import_module(f"{package_root}.giga_models")


__all__ = ["install_aliases"]
