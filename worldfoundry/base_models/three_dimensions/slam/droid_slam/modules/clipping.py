"""Module for base_models -> three_dimensions -> slam -> droid_slam -> modules -> clipping.py functionality."""

import torch
import torch.nn as nn
import torch.nn.functional as F

GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    """Grad clip implementation."""
    @staticmethod
    def forward(ctx, x):
        """Forward.

        Args:
            ctx: The ctx.
            x: The x.
        """
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """Backward.

        Args:
            ctx: The ctx.
            grad_x: The grad x.
        """
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    """Gradient clip implementation."""
    def __init__(self):
        """Init."""
        super(GradientClip, self).__init__()

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return GradClip.apply(x)
