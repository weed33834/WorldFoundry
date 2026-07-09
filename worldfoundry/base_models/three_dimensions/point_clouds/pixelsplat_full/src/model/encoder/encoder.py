"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> model -> encoder -> encoder.py functionality."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    """Encoder implementation."""
    cfg: T

    def __init__(self, cfg: T) -> None:
        """Init.

        Args:
            cfg: The cfg.

        Returns:
            The return value.
        """
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> Gaussians:
        """Forward.

        Args:
            context: The context.
            deterministic: The deterministic.

        Returns:
            The return value.
        """
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
