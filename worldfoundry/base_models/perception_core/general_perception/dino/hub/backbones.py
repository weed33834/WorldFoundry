"""DINO v1 backbone constructors (facebookresearch/dino hub API)."""

from __future__ import annotations

from .. import models as vits


def dino_vits16(pretrained: bool = False, **kwargs):
    """DINO ViT-S/16."""
    model = vits.vit_small(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("Local DINO hub expects callers to provide checkpoint weights.")
    return model


def dino_vits8(pretrained: bool = False, **kwargs):
    """DINO ViT-S/8."""
    model = vits.vit_small(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("Local DINO hub expects callers to provide checkpoint weights.")
    return model


def dino_vitb16(pretrained: bool = False, **kwargs):
    """DINO ViT-B/16."""
    model = vits.vit_base(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("Local DINO hub expects callers to provide checkpoint weights.")
    return model


def dino_vitb8(pretrained: bool = False, **kwargs):
    """DINO ViT-B/8."""
    model = vits.vit_base(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        raise RuntimeError("Local DINO hub expects callers to provide checkpoint weights.")
    return model


def dino_resnet50(pretrained: bool = False, **kwargs):
    """DINO ResNet-50 backbone with identity classifier head."""
    import torch
    from torchvision.models.resnet import resnet50

    model = resnet50(weights=None, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        raise RuntimeError("Local DINO hub expects callers to provide checkpoint weights.")
    return model
