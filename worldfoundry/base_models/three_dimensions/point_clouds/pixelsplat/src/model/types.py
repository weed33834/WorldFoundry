"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> model -> types.py functionality."""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    """Gaussians implementation."""
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Tensor | None = None
    rotations: Tensor | None = None
