"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> loss -> loss_mse.py functionality."""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    """Loss mse cfg implementation."""
    weight: float


@dataclass
class LossMseCfgWrapper:
    """Loss mse cfg wrapper implementation."""
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    """Loss mse implementation."""
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
        delta = prediction.color - batch["target"]["image"]
        return self.cfg.weight * (delta**2).mean()
