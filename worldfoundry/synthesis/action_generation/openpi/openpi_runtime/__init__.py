from __future__ import annotations

import importlib
import sys


def install_aliases() -> None:
    """Register upstream package names for the vendored OpenPI runtime."""
    package_root = __name__
    for public_name in ("openpi", "openpi_client"):
        if public_name not in sys.modules:
            sys.modules[public_name] = importlib.import_module(f"{package_root}.{public_name}")


__all__ = ["install_aliases"]
