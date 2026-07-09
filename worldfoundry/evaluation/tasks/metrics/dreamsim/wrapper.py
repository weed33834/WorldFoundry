"""WorldFoundry facade for DreamSim perceptual image similarity."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
ImageInput = Union[str, Path, Image.Image, np.ndarray]


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_dreamsim() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


@lru_cache(maxsize=4)
def _load_dreamsim_model(
    cache_dir: str,
    dreamsim_type: str,
    device: str,
) -> tuple[Any, Any]:
    _ensure_dreamsim()
    from dreamsim.model import dreamsim

    model, preprocess = dreamsim(
        pretrained=True,
        cache_dir=cache_dir,
        device=device,
        dreamsim_type=dreamsim_type,
    )
    return model, preprocess


def compute_dreamsim(
    reference: ImageInput,
    generated: ImageInput,
    *,
    cache_dir: str | Path | None = None,
    dreamsim_type: str = "ensemble",
    device: str | None = None,
) -> float:
    """Compute DreamSim perceptual distance between two images (lower is more similar)."""
    device_t = _resolve_device(device)
    model_dir = str(
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path("cache/hfd") / "dreamsim"
    )
    model, preprocess = _load_dreamsim_model(model_dir, dreamsim_type, str(device_t))
    ref = preprocess(_load_image(reference)).to(device_t)
    gen = preprocess(_load_image(generated)).to(device_t)
    with torch.no_grad():
        return float(model(ref, gen).item())


__all__ = [
    "compute_dreamsim",
    "package_root",
]
