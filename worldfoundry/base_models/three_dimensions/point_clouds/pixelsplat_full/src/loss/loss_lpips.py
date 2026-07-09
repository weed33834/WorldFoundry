"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> loss -> loss_lpips.py functionality."""

from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    """Loss lpips cfg implementation."""
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    """Loss lpips cfg wrapper implementation."""
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    """Loss lpips implementation."""
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        """Init.

        Args:
            cfg: The cfg.

        Returns:
            The return value.
        """
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

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
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
