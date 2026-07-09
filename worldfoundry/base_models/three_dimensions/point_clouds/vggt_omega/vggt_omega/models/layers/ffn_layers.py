# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> three_dimensions -> point_clouds -> vggt_omega -> vggt_omega -> models -> layers -> ffn_layers.py functionality."""

from typing import Callable, List, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from .utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    """List forward mixin implementation."""
    def forward(self, x: Tensor):
        """Forward.

        Args:
            x: The x.
        """
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        """Forward list.

        Args:
            x_list: The x list.

        Returns:
            The return value.
        """
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    """Mlp implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            drop: The drop.
            bias: The bias.
            device: The device.

        Returns:
            The return value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    """Swi gluffn implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            drop: The drop.
            bias: The bias.
            align_to: The align to.
            device: The device.

        Returns:
            The return value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
