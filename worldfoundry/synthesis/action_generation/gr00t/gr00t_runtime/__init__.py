from __future__ import annotations

import importlib
import sys


def install_aliases() -> None:
    """Register the upstream gr00t package name for the vendored runtime."""
    package_root = __name__
    if "gr00t" not in sys.modules:
        sys.modules["gr00t"] = importlib.import_module(package_root)


__all__ = ["install_aliases"]
