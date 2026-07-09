"""Module for base_models -> diffusion_model -> video -> vchitect -> vchitect_runtime -> models -> utils.py functionality."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """Helper function to norm.

        Args:
            x: The x.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x: The x.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.ones_(self.weight)  # type: ignore
