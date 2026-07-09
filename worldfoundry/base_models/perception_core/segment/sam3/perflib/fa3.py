# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Module for base_models -> perception_core -> segment -> sam3 -> perflib -> fa3.py functionality."""

import torch


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Flash attn func op.

    Args:
        q: The q.
        k: The k.
        v: The v.

    Returns:
        The return value.
    """
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


def flash_attn_func(q, k, v):
    """Flash attn func.

    Args:
        q: The q.
        k: The k.
        v: The v.
    """
    dtype = torch.float8_e4m3fn
    return flash_attn_func_op(q.to(dtype), k.to(dtype), v.to(dtype)).to(q.dtype)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    """Helper function to _.

    Args:
        q: The q.
        k: The k.
        v: The v.
    """
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    # output needs to be bfloat16, not float8!
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
