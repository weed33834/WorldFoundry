"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> model -> encoder -> backbone -> __init__.py functionality."""

from typing import Any

from .backbone import Backbone
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    """Get backbone.

    Args:
        cfg: The cfg.
        d_in: The d in.

    Returns:
        The return value.
    """
    return BACKBONES[cfg.name](cfg, d_in)
