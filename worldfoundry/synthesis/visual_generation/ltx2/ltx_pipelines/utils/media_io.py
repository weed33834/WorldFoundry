import math

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """Resize an image tensor while preserving aspect ratio."""
    height, width = image.shape[-3:-1]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize preserving aspect ratio, then center crop to exact dimensions."""
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]
    return rearrange(tensor, "f c h w -> 1 c f h w")


def normalize_images(images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (images / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_image_and_preprocess(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int = DEFAULT_IMAGE_CRF,
) -> torch.Tensor:
    image = decode_image(image_path=image_path)
    image = preprocess(image=image, crf=crf)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    return normalize_images(image, device, dtype)


def decode_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def preprocess(image: np.ndarray, crf: float = DEFAULT_IMAGE_CRF) -> np.ndarray:
    """Round-trip through JPEG to match LTX conditioning preprocessing."""
    import io

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=crf)
    buffer.seek(0)
    return np.array(Image.open(buffer))
