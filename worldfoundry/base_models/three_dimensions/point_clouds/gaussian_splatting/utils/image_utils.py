#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Module for base_models -> three_dimensions -> point_clouds -> gaussian_splatting -> utils -> image_utils.py functionality."""

import torch

def mse(img1, img2):
    """Mse.

    Args:
        img1: The img1.
        img2: The img2.
    """
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """Psnr.

    Args:
        img1: The img1.
        img2: The img2.
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
