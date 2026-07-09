"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> loss -> loss.py functionality."""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    """Loss implementation."""
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        """Init.

        Args:
            cfg: The cfg.

        Returns:
            The return value.
        """
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
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
        pass
