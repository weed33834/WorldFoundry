"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> third_party -> dino -> hubconf.py functionality."""

from __future__ import annotations

import torch
from torchvision.models.resnet import resnet50

import vision_transformer as vits

dependencies = ["torch", "torchvision"]


def dino_vits16(pretrained: bool = False, **kwargs):
    """Dino vits16.

    Args:
        pretrained: The pretrained.
    """
    model = vits.vit_small(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("WorldFoundry local DINO hub expects pixelSplat checkpoints to provide weights.")
    return model


def dino_vits8(pretrained: bool = False, **kwargs):
    """Dino vits8.

    Args:
        pretrained: The pretrained.
    """
    model = vits.vit_small(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("WorldFoundry local DINO hub expects pixelSplat checkpoints to provide weights.")
    return model


def dino_vitb16(pretrained: bool = False, **kwargs):
    """Dino vitb16.

    Args:
        pretrained: The pretrained.
    """
    model = vits.vit_base(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("WorldFoundry local DINO hub expects pixelSplat checkpoints to provide weights.")
    return model


def dino_vitb8(pretrained: bool = False, **kwargs):
    """Dino vitb8.

    Args:
        pretrained: The pretrained.
    """
    model = vits.vit_base(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("WorldFoundry local DINO hub expects pixelSplat checkpoints to provide weights.")
    return model


def dino_resnet50(pretrained: bool = False, **kwargs):
    """Dino resnet50.

    Args:
        pretrained: The pretrained.
    """
    model = resnet50(weights=None, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        raise RuntimeError("WorldFoundry local DINO hub expects pixelSplat checkpoints to provide weights.")
    return model
