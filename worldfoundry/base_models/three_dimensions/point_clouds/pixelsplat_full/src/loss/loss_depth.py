"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> loss -> loss_depth.py functionality."""

from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    """Loss depth cfg implementation."""
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    """Loss depth cfg wrapper implementation."""
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    """Loss depth implementation."""
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        """Forward.

        Args:
            prediction: The prediction.
            batch: The batch.
            gaussians: The gaussians.
            global_step: The global step.

        Returns:
            The return value.
        """
        # Scale the depth between the near and far planes.
        near = batch["target"]["near"][..., None, None].log()
        far = batch["target"]["far"][..., None, None].log()
        depth = prediction.depth.minimum(far).maximum(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.cfg.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.cfg.sigma_image is not None:
            color_gt = batch["target"]["image"]
            color_dx = reduce(color_gt.diff(dim=-1), "b v c h w -> b v h w", "max")
            color_dy = reduce(color_gt.diff(dim=-2), "b v c h w -> b v h w", "max")
            if self.cfg.use_second_derivative:
                color_dx = color_dx[..., :, 1:].maximum(color_dx[..., :, :-1])
                color_dy = color_dy[..., 1:, :].maximum(color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(-color_dx * self.cfg.sigma_image)
            depth_dy = depth_dy * torch.exp(-color_dy * self.cfg.sigma_image)

        return self.cfg.weight * (depth_dx.abs().mean() + depth_dy.abs().mean())
