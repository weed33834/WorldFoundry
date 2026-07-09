from __future__ import annotations

import math

import numpy as np
from PIL import Image


def align_to(value: int | float, alignment: int | float) -> int:
    return int(math.ceil(float(value) / float(alignment)) * int(alignment))


def align_floor_to(value: int | float, alignment: int | float) -> int:
    return int(math.floor(float(value) / float(alignment)) * int(alignment))


def black_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (int(width), int(height)), (0, 0, 0))


def get_closest_ratio(height: float, width: float, ratios: np.ndarray, buckets: list[tuple[int, int]]):
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio
    if aspect_ratio >= 1:
        indices = [(index, value) for index, value in enumerate(diff_ratios) if value <= 0]
    else:
        indices = [(index, value) for index, value in enumerate(diff_ratios) if value > 0]
    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    return buckets[closest_ratio_id], ratios[closest_ratio_id]


def generate_crop_size_list(base_size: int = 256, patch_size: int = 32, max_ratio: float = 4.0) -> list[tuple[int, int]]:
    num_patches = round((base_size / patch_size) ** 2)
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be >= 1.0")
    crop_size_list: list[tuple[int, int]] = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list
