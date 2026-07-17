"""Color conversions used when composing and exporting Gaussians."""

from __future__ import annotations

import torch


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    nonlinear = ((value + 0.055) / 1.055).clamp_min(0.0).pow(2.4)
    return torch.where(value <= 0.04045, value / 12.92, nonlinear)


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    nonlinear = 1.055 * value.clamp_min(0.0).pow(1.0 / 2.4) - 0.055
    return torch.where(value <= 0.0031308, value * 12.92, nonlinear)
