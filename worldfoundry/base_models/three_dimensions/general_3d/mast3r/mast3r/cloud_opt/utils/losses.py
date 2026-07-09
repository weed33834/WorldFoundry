# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# losses for sparse ga
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> mast3r -> cloud_opt -> utils -> losses.py functionality."""

import torch
import numpy as np


def l05_loss(x, y):
    """L05 loss.

    Args:
        x: The x.
        y: The y.
    """
    return torch.linalg.norm(x - y, dim=-1).sqrt()


def l1_loss(x, y):
    """L1 loss.

    Args:
        x: The x.
        y: The y.
    """
    return torch.linalg.norm(x - y, dim=-1)


def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
    """Gamma loss.

    Args:
        gamma: The gamma.
        mul: The mul.
        offset: The offset.
        clip: The clip.
    """
    if offset is None:
        if gamma == 1:
            return l1_loss
        # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
        offset = (1 / gamma)**(1 / (gamma - 1))

    def loss_func(x, y):
        """Loss func.

        Args:
            x: The x.
            y: The y.
        """
        return (mul * l1_loss(x, y).clip(max=clip) + offset) ** gamma - offset ** gamma
    return loss_func


def meta_gamma_loss():
    """Meta gamma loss."""
    return lambda alpha: gamma_loss(alpha)
