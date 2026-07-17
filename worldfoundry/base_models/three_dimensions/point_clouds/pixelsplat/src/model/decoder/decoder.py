"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> model -> decoder -> decoder.py functionality."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ..types import Gaussians

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    """Decoder output implementation."""
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    """Decoder implementation."""
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        """Init.

        Args:
            cfg: The cfg.
            dataset_cfg: The dataset cfg.

        Returns:
            The return value.
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        """Forward.

        Args:
            gaussians: The gaussians.
            extrinsics: The extrinsics.
            intrinsics: The intrinsics.
            near: The near.
            far: The far.
            image_shape: The image shape.
            depth_mode: The depth mode.

        Returns:
            The return value.
        """
        pass
