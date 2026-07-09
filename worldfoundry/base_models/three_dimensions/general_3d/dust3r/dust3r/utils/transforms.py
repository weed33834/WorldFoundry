# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
"""Module for base_models -> three_dimensions -> general_3d -> dust3r -> dust3r -> utils -> transforms.py functionality."""

try:
    from dust3r.utils.image import ImgNorm
except ModuleNotFoundError as exc:
    if exc.name != "torchvision":
        raise
    import numpy as np
    import torch

    def ImgNorm(image):
        """Imgnorm.

        Args:
            image: The image.
        """
        array = np.asarray(image).astype("float32") / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

__all__ = ["ImgNorm"]
