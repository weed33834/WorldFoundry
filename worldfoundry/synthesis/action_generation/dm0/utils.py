"""Attention and diffusion helpers for checkpoint-compatible DM0 inference."""

from __future__ import annotations

import math

import torch


def make_attn_mask_2d(
    padding_mask: torch.BoolTensor,
    attn_mask: torch.IntTensor,
) -> torch.BoolTensor:
    """Build the block-causal mask used by the policy prefix and action expert."""

    if padding_mask.ndim != 2 or attn_mask.ndim != 2:
        raise ValueError("padding_mask and attn_mask must both be rank two")
    cumulative = torch.cumsum(attn_mask, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = padding_mask[:, None, :] & padding_mask[:, :, None]
    return causal & valid


def make_suffix_attn_mask_2d(
    suffix_padding_mask: torch.BoolTensor,
    suffix_attn_mask: torch.IntTensor,
    prefix_padding_mask: torch.BoolTensor,
    prefix_attn_mask: torch.IntTensor,
) -> torch.BoolTensor:
    suffix_length = suffix_attn_mask.shape[1]
    padding = torch.cat((prefix_padding_mask, suffix_padding_mask), dim=1)
    attention = torch.cat((prefix_attn_mask, suffix_attn_mask), dim=1)
    return make_attn_mask_2d(padding, attention)[:, -suffix_length:, :]


def make_attn_mask_4d(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
    minimum = torch.finfo(dtype).min
    return torch.where(mask, 0.0, minimum)[:, None, :, :].to(dtype=dtype)


def posemb_sincos(
    time: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if dim % 2:
        raise ValueError("the time embedding dimension must be even")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    work_dtype = torch.float64 if time.device.type != "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=work_dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    phase = (2.0 * math.pi / period)[None, :] * time[:, None].to(work_dtype)
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
