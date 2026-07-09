"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> dataset -> view_sampler -> three_view_hack.py functionality."""

import torch
from jaxtyping import Int
from torch import Tensor


def add_third_context_index(
    indices: Int[Tensor, "*batch 2"]
) -> Int[Tensor, "*batch 3"]:
    """Add third context index.

    Args:
        indices: The indices.

    Returns:
        The return value.
    """
    left, right = indices.unbind(dim=-1)
    return torch.stack((left, (left + right) // 2, right), dim=-1)
