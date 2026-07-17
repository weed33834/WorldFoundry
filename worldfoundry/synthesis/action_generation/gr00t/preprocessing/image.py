# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic image preprocessing used by GR00T inference."""

from collections.abc import Sequence

import torch
import torchvision.transforms.v2 as transforms


class LetterBoxTransform:
    """Pad image tensors to a square without changing their aspect ratio."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = transforms.functional.to_image(image)
        height, width = image.shape[-2:]
        if height == width:
            return image

        size = max(height, width)
        pad_height = size - height
        pad_width = size - width
        left = pad_width // 2
        top = pad_height // 2
        right = pad_width - left
        bottom = pad_height - top
        return transforms.functional.pad(
            image,
            padding=[left, top, right, bottom],
            fill=0,
        )


def _pair(value: Sequence[int] | int | None, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a two-dimensional image size, got {value!r}")
    return int(value[0]), int(value[1])


def build_image_transformations(
    image_target_size: Sequence[int] | int | None,
    image_crop_size: Sequence[int] | int | None,
) -> transforms.Compose:
    """Build the checkpoint-compatible deterministic evaluation transform.

    Inference uses the checkpoint evaluation path: convert to an image tensor,
    letterbox to square, resize, center-crop, and resize to the model resolution.
    The v2 transforms preserve arbitrary leading batch/time dimensions.
    """

    target_size = _pair(image_target_size, (256, 256))
    crop_size = _pair(image_crop_size, (230, 230))
    return transforms.Compose(
        [
            transforms.ToImage(),
            LetterBoxTransform(),
            transforms.Resize(size=target_size, antialias=True),
            transforms.CenterCrop(size=crop_size),
            transforms.Resize(size=target_size, antialias=True),
        ]
    )
