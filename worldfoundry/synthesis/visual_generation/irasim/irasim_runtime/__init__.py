"""Vendored IRASim runtime package."""

from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent


def ensure_legacy_import_paths() -> str:
    """Expose IRASim's upstream absolute-import layout without eager-loading subpackages."""
    path = str(RUNTIME_ROOT)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    return path


ensure_legacy_import_paths()


__all__ = ["RUNTIME_ROOT", "ensure_legacy_import_paths"]
