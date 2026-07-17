"""Unified pixelSplat source, runtime, and shared components."""

from __future__ import annotations

import sys


def ensure_pixelsplat_source_alias() -> object:
    """Register the upstream ``src`` package alias on demand."""
    from . import src as _src

    sys.modules.setdefault("src", _src)
    return _src


ensure_pixelsplat_source_alias()


from .worldfoundry_runtime import (
    DEFAULT_PIXELSPLAT_ASSET_ROOT,
    DEFAULT_PIXELSPLAT_CKPT,
    DEFAULT_PIXELSPLAT_DEMO_INDEX,
    DEFAULT_PIXELSPLAT_DEMO_ROOT,
    DEFAULT_PIXELSPLAT_INDEX,
    PixelSplatRuntime,
)

__all__ = [
    "DEFAULT_PIXELSPLAT_ASSET_ROOT",
    "DEFAULT_PIXELSPLAT_CKPT",
    "DEFAULT_PIXELSPLAT_DEMO_INDEX",
    "DEFAULT_PIXELSPLAT_DEMO_ROOT",
    "DEFAULT_PIXELSPLAT_INDEX",
    "PixelSplatRuntime",
    "cuda_splatting",
    "decoder_splatting_cuda",
    "ensure_pixelsplat_source_alias",
    "projection",
]
