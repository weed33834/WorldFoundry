"""Module for base_models -> three_dimensions -> depth -> midas -> backbones -> next_vit.py functionality."""

import timm

import torch.nn as nn

from pathlib import Path
from .utils import activations, forward_default, get_activation

from ..external.next_vit.classification.nextvit import *


def forward_next_vit(pretrained, x):
    """Forward next vit.

    Args:
        pretrained: The pretrained.
        x: The x.
    """
    return forward_default(pretrained, x, "forward")


def _make_next_vit_backbone(
        model,
        hooks=[2, 6, 36, 39],
):
    """Helper function to make next vit backbone.

    Args:
        model: The model.
        hooks: The hooks.
    """
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.features[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.features[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.features[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.features[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    return pretrained


def _make_pretrained_next_vit_large_6m(hooks=None):
    """Helper function to make pretrained next vit large 6m.

    Args:
        hooks: The hooks.
    """
    model = timm.create_model("nextvit_large")

    hooks = [2, 6, 36, 39] if hooks == None else hooks
    return _make_next_vit_backbone(
        model,
        hooks=hooks,
    )
