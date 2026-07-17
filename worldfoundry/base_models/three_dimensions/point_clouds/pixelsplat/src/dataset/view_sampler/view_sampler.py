"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> dataset -> view_sampler -> view_sampler.py functionality."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ..types import Stage

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    """View sampler implementation."""
    cfg: T
    stage: Stage
    is_overfitting: bool
    cameras_are_circular: bool

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
    ) -> None:
        """Init.

        Args:
            cfg: The cfg.
            stage: The stage.
            is_overfitting: The is overfitting.
            cameras_are_circular: The cameras are circular.

        Returns:
            The return value.
        """
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular

    @abstractmethod
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Sample.

        Args:
            scene: The scene.
            extrinsics: The extrinsics.
            intrinsics: The intrinsics.
            device: The device.

        Returns:
            The return value.
        """
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        """Num target views.

        Returns:
            The return value.
        """
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        """Num context views.

        Returns:
            The return value.
        """
        pass
