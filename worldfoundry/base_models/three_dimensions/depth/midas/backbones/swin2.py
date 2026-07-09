"""Module for base_models -> three_dimensions -> depth -> midas -> backbones -> swin2.py functionality."""

import timm

from .swin_common import _make_swin_backbone


def _make_pretrained_swin2l24_384(pretrained, hooks=None):
    """Helper function to make pretrained swin2l24 384.

    Args:
        pretrained: The pretrained.
        hooks: The hooks.
    """
    model = timm.create_model("swinv2_large_window12to24_192to384_22kft1k", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2b24_384(pretrained, hooks=None):
    """Helper function to make pretrained swin2b24 384.

    Args:
        pretrained: The pretrained.
        hooks: The hooks.
    """
    model = timm.create_model("swinv2_base_window12to24_192to384_22kft1k", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2t16_256(pretrained, hooks=None):
    """Helper function to make pretrained swin2t16 256.

    Args:
        pretrained: The pretrained.
        hooks: The hooks.
    """
    model = timm.create_model("swinv2_tiny_window16_256", pretrained=pretrained)

    hooks = [1, 1, 5, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks,
        patch_grid=[64, 64]
    )
