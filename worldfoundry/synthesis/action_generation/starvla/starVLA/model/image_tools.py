"""Image helpers required by the vendored StarVLA inference modules."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def to_pil_preserve(images: Any, scale_float: bool = True):
    """Convert images to PIL leaves while preserving nested list/tuple structure."""

    def convert(obj):
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(convert(item) for item in obj)
        if isinstance(obj, Image.Image):
            return obj
        if not isinstance(obj, np.ndarray):
            raise TypeError(f"Unsupported element type: {type(obj)}")

        arr = obj
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got shape={arr.shape}")
        if arr.shape[2] not in (1, 3, 4):
            raise ValueError(f"Channel count must be 1/3/4, got {arr.shape[2]}")
        if np.issubdtype(arr.dtype, np.floating):
            if not scale_float:
                raise TypeError("Float array provided but scale_float=False")
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        return Image.fromarray(arr, mode="RGB" if arr.shape[2] == 3 else "RGBA")

    return convert(images)
