# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> layers -> dino_head.py functionality."""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class DINOHead(nn.Module):
    """Dino head implementation."""
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
            use_bn: The use bn.
            nlayers: The nlayers.
            hidden_dim: The hidden dim.
            bottleneck_dim: The bottleneck dim.
            mlp_bias: The mlp bias.
        """
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        """Init weights.

        Returns:
            The return value.
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper function to init weights.

        Args:
            m: The m.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        """Forward.

        Args:
            x: The x.
            no_last_layer: The no last layer.
            only_last_layer: The only last layer.
        """
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """Helper function to build mlp.

    Args:
        nlayers: The nlayers.
        in_dim: The in dim.
        bottleneck_dim: The bottleneck dim.
        hidden_dim: The hidden dim.
        use_bn: The use bn.
        bias: The bias.
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
