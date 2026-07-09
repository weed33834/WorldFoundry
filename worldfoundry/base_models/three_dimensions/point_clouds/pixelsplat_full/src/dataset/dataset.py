"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> dataset -> dataset.py functionality."""

from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    """Dataset cfg common implementation."""
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
