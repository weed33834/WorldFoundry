"""Module for base_models -> diffusion_model -> video -> vchitect -> vchitect_runtime -> op_replace.py functionality."""

from __future__ import annotations

import importlib.util

import torch


def _module_exists(module_name: str) -> bool:
    """Check whether a module can be imported without loading it.

    Args:
        module_name: Fully qualified module name to probe.
    """
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        if importlib.util.find_spec(".".join(parts[:index])) is None:
            return False
    return True


def _load_fused_layer_norm():
    """Load the optional fused LayerNorm implementation.

    Args:
        None: The function probes installed optional packages in the active environment.
    """
    if _module_exists("apex.normalization"):
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm
        except ImportError:
            pass
    if _module_exists("xformers.triton"):
        try:
            from xformers.triton import FusedLayerNorm

            return FusedLayerNorm
        except ImportError:
            pass
    return None


def replace_all_layernorms(model):
    """Replace torch LayerNorm modules with an available fused implementation.

    Args:
        model: Module tree whose child LayerNorm modules should be replaced in place.
    """
    fused_layer_norm = _load_fused_layer_norm()
    if fused_layer_norm is None:
        print("WARNING: apex.normalization and xformers.triton.FusedLayerNorm are not found, skip using FusedLayerNorm")
        return model
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LayerNorm):
            setattr(model, name, fused_layer_norm(module.normalized_shape, module.eps, module.elementwise_affine))
        else:
            replace_all_layernorms(module)
    return model


def replace_all_groupnorms(model):
    """Keep GroupNorm modules unchanged when apex group norm is unavailable.

    Args:
        model: Module tree whose GroupNorm modules are inspected.
    """
    if not _module_exists("apex.contrib.group_norm"):
        print("WARNING: apex.contrib.group_norm is not found, skip using apex groupnorm")
        return model
    from apex.contrib.group_norm import GroupNorm

    for name, module in model.named_children():
        if isinstance(module, torch.nn.GroupNorm):
            setattr(model, name, GroupNorm(module.num_groups, module.num_channels, eps=module.eps, affine=module.affine))
        else:
            replace_all_groupnorms(module)
    return model
