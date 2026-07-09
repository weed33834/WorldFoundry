# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

"""Module for base_models -> three_dimensions -> point_clouds -> hunyuan_mirror -> models -> layers -> attention.py functionality."""

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    """Attention implementation."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use _worldfoundry_scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            qkv_bias: The qkv bias.
            proj_bias: The proj bias.
            attn_drop: The attn drop.
            proj_drop: The proj drop.
            norm_layer: The norm layer.
            qk_norm: The qk norm.
            fused_attn: The fused attn.
            rope: The rope.

        Returns:
            The return value.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            pos: The pos.

        Returns:
            The return value.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # orig_dtype = q.dtype
        x = _worldfoundry_scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        # if x.dtype != orig_dtype:
        #     x = x.to(orig_dtype)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """Mem eff attention implementation."""
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            attn_bias: The attn bias.
            pos: The pos.

        Returns:
            The return value.
        """
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
