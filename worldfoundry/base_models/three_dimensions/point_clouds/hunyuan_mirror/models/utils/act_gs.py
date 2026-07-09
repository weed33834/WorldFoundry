"""Module for base_models -> three_dimensions -> point_clouds -> hunyuan_mirror -> models -> utils -> act_gs.py functionality."""

import torch
from einops import rearrange

    
def reg_dense_offsets(xyz, shift=6.0):
    """Reg dense offsets.

    Args:
        xyz: The xyz.
        shift: The shift.
    """
    d = xyz.norm(dim=-1, keepdim=True)
    return xyz / d.clamp(min=1e-8) * (torch.exp(d - shift) - torch.exp(-shift))

def reg_dense_scales(scales):
    """Reg dense scales.

    Args:
        scales: The scales.
    """
    return scales.exp()

def reg_dense_rotation(rotations, eps=1e-8):
    """Reg dense rotation.

    Args:
        rotations: The rotations.
        eps: The eps.
    """
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

def reg_dense_sh(sh):
    """Reg dense sh.

    Args:
        sh: The sh.
    """
    return rearrange(sh, '... (d_sh xyz) -> ... d_sh xyz', xyz=3)

def reg_dense_opacities(opacities):
    """Reg dense opacities.

    Args:
        opacities: The opacities.
    """
    return opacities.sigmoid()

def reg_dense_weights(weights):
    """Reg dense weights.

    Args:
        weights: The weights.
    """
    return weights.sigmoid()
