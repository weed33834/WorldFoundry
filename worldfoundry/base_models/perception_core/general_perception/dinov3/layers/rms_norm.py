# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> layers -> rms_norm.py functionality."""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Rms norm implementation."""
    def __init__(self, dim: int, eps: float = 1e-5):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters.

        Returns:
            The return value.
        """
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        """Helper function to norm.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
