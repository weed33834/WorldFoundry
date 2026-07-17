"""Minimal tensor helpers required by Uni3C point-cloud inference."""

from __future__ import annotations

import torch


def points_padding(points: torch.Tensor) -> torch.Tensor:
    padding = torch.ones_like(points)[..., :1]
    return torch.cat([points, padding], dim=-1)


__all__ = ["points_padding"]
