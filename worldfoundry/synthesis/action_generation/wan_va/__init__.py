"""LingBot-VA Wan-VA foundation source package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
WAN_VA_ROOT = RUNTIME_ROOT / "wan_va"
WAN_VA_PACKAGE = "worldfoundry.synthesis.action_generation.wan_va.wan_va"


def install_aliases() -> None:
    """Register upstream Wan-VA package aliases for vendored imports."""

    package = importlib.import_module(WAN_VA_PACKAGE)
    package_path = str(WAN_VA_ROOT)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    sys.modules.setdefault("wan_va", package)


__all__ = ["RUNTIME_ROOT", "WAN_VA_PACKAGE", "WAN_VA_ROOT", "install_aliases"]
