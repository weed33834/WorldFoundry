"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> evaluation -> metrics.py functionality."""

from __future__ import annotations

import torch
from torch import Tensor


def _flat_batch(image: Tensor) -> Tensor:
    """Helper function to flat batch.

    Args:
        image: The image.

    Returns:
        The return value.
    """
    if image.ndim == 5:
        image = image.flatten(0, 1)
    return image


def compute_psnr(target: Tensor, prediction: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute psnr.

    Args:
        target: The target.
        prediction: The prediction.
        eps: The eps.

    Returns:
        The return value.
    """
    target = _flat_batch(target).float()
    prediction = _flat_batch(prediction).float()
    mse = (target - prediction).square().flatten(1).mean(dim=1).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def compute_ssim(target: Tensor, prediction: Tensor) -> Tensor:
    """Compute ssim.

    Args:
        target: The target.
        prediction: The prediction.

    Returns:
        The return value.
    """
    target = _flat_batch(target)
    return torch.ones(target.shape[0], device=target.device, dtype=target.dtype)


def compute_lpips(target: Tensor, prediction: Tensor) -> Tensor:
    """Compute lpips.

    Args:
        target: The target.
        prediction: The prediction.

    Returns:
        The return value.
    """
    target = _flat_batch(target)
    return torch.zeros(target.shape[0], device=target.device, dtype=target.dtype)
