from __future__ import annotations

import importlib
import sys


RUNTIME_PACKAGE = __name__


def install_runtime_aliases() -> None:
    """Expose official DreamZero package names from the in-tree runtime.

    Args:
        None.
    """
    for alias in ("groot", "eval_utils"):
        target = importlib.import_module(f"{RUNTIME_PACKAGE}.{alias}")
        sys.modules.setdefault(alias, target)


install_runtime_aliases()

__all__ = ["RUNTIME_PACKAGE", "install_runtime_aliases"]
