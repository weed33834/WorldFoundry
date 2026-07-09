"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> model -> encoder -> backbone -> backbone.py functionality."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ....dataset.types import BatchedViews

T = TypeVar("T")


class Backbone(nn.Module, ABC, Generic[T]):
    """Backbone implementation."""
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
    ) -> Float[Tensor, "batch view d_out height width"]:
        """Forward.

        Args:
            context: The context.

        Returns:
            The return value.
        """
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        """D out.

        Returns:
            The return value.
        """
        pass
