from __future__ import annotations

import importlib
import sys


def install_aliases() -> None:
    """Register upstream package names for the vendored LAPA runtime."""
    package_root = f"{__name__}.latent_runtime"
    module = importlib.import_module(package_root)
    if "latent_runtime" not in sys.modules:
        sys.modules["latent_runtime"] = module


__all__ = ["install_aliases"]
