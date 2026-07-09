"""pixelSplat runtime package with source promoted to base_models."""

from __future__ import annotations

import sys


def ensure_pixelsplat_source_alias() -> object:
    """Register pixelSplat's upstream ``src`` package alias on demand."""
    from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat_full import src as _src

    sys.modules.setdefault("src", _src)
    return _src

from .worldfoundry_runtime import (
    DEFAULT_PIXELSPLAT_ASSET_ROOT,
    DEFAULT_PIXELSPLAT_CKPT,
    DEFAULT_PIXELSPLAT_INDEX,
    PixelSplatRuntime,
)

__all__ = [
    "DEFAULT_PIXELSPLAT_ASSET_ROOT",
    "DEFAULT_PIXELSPLAT_CKPT",
    "DEFAULT_PIXELSPLAT_INDEX",
    "PixelSplatRuntime",
    "ensure_pixelsplat_source_alias",
]
