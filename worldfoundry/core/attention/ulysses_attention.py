# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch.distributed as dist

from worldfoundry.core.attention import flash_attention
from worldfoundry.core.distributed.sequence_ops import all_to_all, all_to_all_many


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
    *,
    attention_fn=flash_attention,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    # gather q/k/v sequence
    q, k, v = all_to_all_many(
        (q, k, v),
        scatter_dim=2,
        gather_dim=1,
    )

    # apply attention
    x = attention_fn(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x


def flattened_ulysses_attention(
    q,
    k,
    v,
    heads,
    *,
    attention_fn,
    mask=None,
    group=None,
):
    """Run an LTX-style flattened attention callable with Ulysses exchange.

    ``q``, ``k`` and ``v`` use the common ``[B, S_local, H*D]`` layout.  The
    exchange turns local sequence/full heads into full sequence/local heads,
    calls the supplied exact attention implementation, and exchanges back.
    The underlying callable therefore remains free to select SDPA, cuDNN,
    FlashAttention or the math fallback for the current GPU generation.

    A full-sequence additive mask can be supplied for padding or spatial-memory
    key validity.  Cross attention should not use this helper because its text
    key/value sequence is already replicated on every rank.
    """

    if not dist.is_available() or not dist.is_initialized():
        if mask is None:
            return attention_fn(q, k, v, heads)
        return attention_fn(q, k, v, heads, mask)

    world_size = dist.get_world_size(group)
    if world_size <= 1:
        if mask is None:
            return attention_fn(q, k, v, heads)
        return attention_fn(q, k, v, heads, mask)
    if heads % world_size:
        raise ValueError(f"attention heads ({heads}) must be divisible by context-parallel size ({world_size})")

    batch, _, hidden = q.shape
    if hidden % heads:
        raise ValueError(f"hidden size ({hidden}) must be divisible by attention heads ({heads})")
    head_dim = hidden // heads
    q, k, v = (
        tensor.view(batch, -1, heads, head_dim)
        for tensor in (q, k, v)
    )
    q, k, v = all_to_all_many(
        (q, k, v),
        scatter_dim=2,
        gather_dim=1,
        group=group,
    )
    local_heads = heads // world_size
    full_q = q.reshape(batch, -1, local_heads * head_dim)
    full_k = k.reshape(batch, -1, local_heads * head_dim)
    full_v = v.reshape(batch, -1, local_heads * head_dim)
    if mask is None:
        output = attention_fn(full_q, full_k, full_v, local_heads)
    else:
        output = attention_fn(full_q, full_k, full_v, local_heads, mask)
    output = output.view(batch, -1, local_heads, head_dim)
    output = all_to_all(output, scatter_dim=1, gather_dim=2, group=group)
    return output.reshape(batch, -1, hidden)


__all__ = ["distributed_attention", "flattened_ulysses_attention"]
