# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> diffusion -> module -> attention.py functionality."""

from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

try:
    import transformer_engine as te
    from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb as te_apply_rotary_pos_emb
except Exception:
    te = None
    DotProductAttention = None
    te_apply_rotary_pos_emb = None

# ---------------------- Feed Forward Network -----------------------


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        """Init.

        Args:
            d_model: The d model.
            d_ff: The d ff.
            dropout: The dropout.
            activation: The activation.
            is_gated: The is gated.
            bias: The bias.

        Returns:
            The return value.
        """
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x: The x.
        """
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    """Feed forward implementation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        """Init.

        Args:
            d_model: The d model.
            d_ff: The d ff.
            dropout: The dropout.
            bias: The bias.
        """
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x: The x.
        """
        assert self.dropout.p == 0.0, "we skip dropout"

        x = self.layer1(x)

        def activation_layer2_forward(x):
            """Activation layer2 forward.

            Args:
                x: The x.
            """
            x = self.activation(x)
            x = self.layer2(x)
            return x

        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


# ---------------------- Normalization Layer -----------------------


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def get_normalization(name: str, channels: int):
    """Get normalization.

    Args:
        name: The name.
        channels: The channels.
    """
    if name == "I":
        return nn.Identity()
    elif name == "R":
        if te is not None and hasattr(te, "pytorch"):
            return te.pytorch.RMSNorm(channels, eps=1e-6)
        return nn.RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


def apply_rotary_pos_emb_torch(x: torch.Tensor, rope_emb: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Apply rotary pos emb torch.

    Args:
        x: The x.
        rope_emb: The rope emb.
        qkv_format: The qkv format.

    Returns:
        The return value.
    """
    if qkv_format == "bshd":
        rope_emb = rearrange(rope_emb, "s b h d -> b s h d")
    elif qkv_format != "sbhd":
        raise ValueError(f"Unsupported qkv_format {qkv_format} for torch RoPE fallback")

    cos = rope_emb.cos().to(dtype=x.dtype, device=x.device)
    sin = rope_emb.sin().to(dtype=x.dtype, device=x.device)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class BaseAttentionOp(nn.Module):
    """Base attention op implementation."""
    def __init__(self):
        """Init."""
        super().__init__()


class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None, self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for query, key, and value projections.
                                  Defaults to "SSI".
        qkv_norm_mode (str, optional): A string representing normalization mode for query, key, and value projections.
                                        Defaults to 'per_head'. Only support 'per_head'.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/cosmos_predict1/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
    ) -> None:
        """Init.

        Args:
            query_dim: The query dim.
            context_dim: The context dim.
            heads: The heads.
            dim_head: The dim head.
            dropout: The dropout.
            attn_op: The attn op.
            qkv_bias: The qkv bias.
            out_bias: The out bias.
            qkv_norm: The qkv norm.
            qkv_norm_mode: The qkv norm mode.
            backend: The backend.
            qkv_format: The qkv format.

        Returns:
            The return value.
        """
        super().__init__()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        self.backend = backend
        self.tp_size = 1  # TP is not included in this Attention implementation.

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

        if attn_op:  # use what is given
            self.attn_op = attn_op
        elif self.backend == "transformer_engine" and DotProductAttention is not None:
            self.attn_op: BaseAttentionOp = DotProductAttention(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
                tp_size=self.tp_size,
                tp_group=None,
                sequence_parallel=False,
            )
        else:
            self.backend = "torch"
            self.attn_op = _worldfoundry_scaled_dot_product_attention
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cal qkv.

        Args:
            x: The x.
            context: The context.
            mask: The mask.
            rope_emb: The rope emb.

        Returns:
            The return value.
        """
        del kwargs

        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        We use a flag `self.qkv_norm_mode` to control the normalization behavior.
        The default value of `self.qkv_norm_mode` is "per_head", which means we normalize per head.
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            q, k, v = map(
                lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head),
                (q, k, v),
            )
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            if self.backend == "transformer_engine" and te_apply_rotary_pos_emb is not None:
                q = te_apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = te_apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            else:
                q = apply_rotary_pos_emb_torch(q, rope_emb, self.qkv_format)
                k = apply_rotary_pos_emb_torch(k, rope_emb, self.qkv_format)
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        """Cal attn.

        Args:
            q: The q.
            k: The k.
            v: The v.
            mask: The mask.
        """
        if self.backend == "transformer_engine":
            seq_dim = self.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
            return self.to_out(out)
        elif self.backend == "torch":
            if self.qkv_format == "bshd":
                q = rearrange(q, "b s h d -> b h s d")
                k = rearrange(k, "b s h d -> b h s d")
                v = rearrange(v, "b s h d -> b h s d")
            elif self.qkv_format == "sbhd":
                q = rearrange(q, "s b h d -> b h s d")
                k = rearrange(k, "s b h d -> b h s d")
                v = rearrange(v, "s b h d -> b h s d")
            else:
                raise ValueError(f"Unsupported qkv_format {self.qkv_format}")
            out = self.attn_op(q, k, v)  # [B, Mq, H, V]
            if self.qkv_format == "bshd":
                out = rearrange(out, "b h s d -> b s (h d)")
            else:
                out = rearrange(out, "b h s d -> s b (h d)")
            return self.to_out(out)
        else:
            raise ValueError(f"Backend {self.backend} not found")

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)
