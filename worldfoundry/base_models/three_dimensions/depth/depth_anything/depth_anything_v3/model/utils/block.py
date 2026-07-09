# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Module for base_models -> three_dimensions -> depth -> depth_anything -> depth_anything_v3 -> model -> utils -> block.py functionality."""

from typing import Callable
from torch import Tensor, nn

from .attention import Attention, LayerScale, Mlp


class Block(nn.Module):
    """Block implementation."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            mlp_ratio: The mlp ratio.
            qkv_bias: The qkv bias.
            proj_bias: The proj bias.
            ffn_bias: The ffn bias.
            drop: The drop.
            attn_drop: The attn drop.
            init_values: The init values.
            drop_path: The drop path.
            act_layer: The act layer.
            norm_layer: The norm layer.
            attn_class: The attn class.
            ffn_layer: The ffn layer.
            qk_norm: The qk norm.
            rope: The rope.

        Returns:
            The return value.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.sample_drop_ratio = 0.0  # Equivalent to always having drop_path=0

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            pos: The pos.
            attn_mask: The attn mask.

        Returns:
            The return value.
        """
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None) -> Tensor:
            """Attn residual func.

            Args:
                x: The x.
                pos: The pos.
                attn_mask: The attn mask.

            Returns:
                The return value.
            """
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            """Ffn residual func.

            Args:
                x: The x.

            Returns:
                The return value.
            """
            return self.ls2(self.mlp(self.norm2(x)))

        # drop_path is always 0, so always take the else branch
        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
        x = x + ffn_residual_func(x)
        return x
