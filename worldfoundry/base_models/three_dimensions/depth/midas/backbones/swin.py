"""Module for base_models -> three_dimensions -> depth -> midas -> backbones -> swin.py functionality."""

import timm

from .swin_common import _make_swin_backbone


def _make_pretrained_swinl12_384(pretrained, hooks=None):
    """Helper function to make pretrained swinl12 384.

    Args:
        pretrained: The pretrained.
        hooks: The hooks.
    """
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )
