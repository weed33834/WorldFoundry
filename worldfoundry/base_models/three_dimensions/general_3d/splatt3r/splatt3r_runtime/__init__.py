"""Inference-only Splatt3R runtime package."""

from __future__ import annotations

import sys


def ensure_runtime_aliases() -> dict[str, object]:
    """Register the shared pixelSplat upstream alias only when requested."""
    from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat import (
        ensure_pixelsplat_source_alias,
    )

    aliases = {
        "src": ensure_pixelsplat_source_alias(),
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)
    return aliases


__all__ = ["ensure_runtime_aliases"]
