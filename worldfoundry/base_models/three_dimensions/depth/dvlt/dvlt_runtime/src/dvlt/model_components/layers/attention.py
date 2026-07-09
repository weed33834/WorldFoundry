# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Originally from DINOv2 (https://github.com/facebookresearch/dinov2),
# distributed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modified for use in dvlt.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Module for base_models -> three_dimensions -> depth -> dvlt -> dvlt_runtime -> src -> dvlt -> model_components -> layers -> attention.py functionality."""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention


try:
    # Flash Attention 3 is Hopper-only; treated as an optional accelerator.
    from flash_attn_interface import flash_attn_func as _fa3_flash_attn_func
except ImportError:
    _fa3_flash_attn_func = None


flex_attention = torch.compile(flex_attention, dynamic=True)

XFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

_attn_state: dict = {"backend": "auto"}


def set_attn_backend(backend: str) -> None:
    """Set the global attention backend used by all ``Attention`` modules.

    Args:
        backend: One of ``"auto"``, ``"flash"``, ``"fa3"``.
    """
    valid = ("auto", "flash", "fa3")
    if backend not in valid:
        raise ValueError(f"attn_backend must be one of {valid}, got '{backend}'")
    if backend == "fa3" and _fa3_flash_attn_func is None:
        raise ImportError(
            "Flash Attention 3 backend selected but `flash_attn_interface` is not installed. "
            "Install it on a Hopper GPU or pick `attn_backend=auto|flash`."
        )
    _attn_state["backend"] = backend
    logger.info(f"Attention backend set to '{backend}'")


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

    def forward(self, x: Tensor, pos=None, block_mask=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            pos: The pos.
            block_mask: The block mask.

        Returns:
            The return value.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if _attn_state["backend"] == "fa3":
            # FA3 expects (B, N, H, D); we have (B, H, N, D) from the permute above
            # Cast q/k back to v's dtype because LayerNorm (q_norm/k_norm) upcasts to fp32
            q, k, v = q.transpose(1, 2).to(v.dtype), k.transpose(1, 2).to(v.dtype), v.transpose(1, 2)
            out = _fa3_flash_attn_func(q, k, v)
            if isinstance(out, tuple):
                out = out[0]
            x = out.transpose(1, 2)
        elif block_mask is not None:
            x = flex_attention(
                q.to(v.dtype),
                k.to(v.dtype),
                v,
                block_mask=block_mask,
            )
        elif self.fused_attn:
            x = _worldfoundry_scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                backend=_attn_state["backend"] if _attn_state["backend"] == "flash" else None,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
