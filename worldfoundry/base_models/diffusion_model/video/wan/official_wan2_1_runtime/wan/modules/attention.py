# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> official_wan2_1_runtime -> wan -> modules -> attention.py functionality."""

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

try:
    import xformers.ops as _xformers_ops
    XFORMERS_AVAILABLE = True
except Exception:
    _xformers_ops = None
    XFORMERS_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        """Half.

        Args:
            x: The x.
        """
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        if FLASH_ATTN_2_AVAILABLE:
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))
        else:
            x = _memory_efficient_varlen_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                batch=b,
                query_length=lq,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
            )

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Attention.

    Args:
        q: The q.
        k: The k.
        v: The v.
        q_lens: The q lens.
        k_lens: The k lens.
        dropout_p: The dropout p.
        softmax_scale: The softmax scale.
        q_scale: The q scale.
        causal: The causal.
        window_size: The window size.
        deterministic: The deterministic.
        dtype: The dtype.
        fa_version: The fa version.
    """
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )


def _memory_efficient_varlen_attention(
    *,
    q,
    k,
    v,
    q_lens,
    k_lens,
    batch,
    query_length,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
):
    """Helper function to memory efficient varlen attention."""
    if window_size != (-1, -1):
        warnings.warn(
            'Sliding-window attention is ignored by the non-flash-attn fallback.'
        )

    outputs = []
    q_start = 0
    k_start = 0
    for q_len, k_len in zip(q_lens.tolist(), k_lens.tolist()):
        q_item = q[q_start:q_start + q_len]
        k_item = k[k_start:k_start + k_len]
        v_item = v[k_start:k_start + k_len]
        if q_item.size(1) != k_item.size(1):
            repeat = q_item.size(1) // k_item.size(1)
            k_item = k_item.repeat_interleave(repeat, dim=1)
            v_item = v_item.repeat_interleave(repeat, dim=1)

        if XFORMERS_AVAILABLE:
            attn_bias = None
            if causal:
                attn_bias = _xformers_ops.LowerTriangularMask()
            out = _xformers_ops.memory_efficient_attention(
                q_item.unsqueeze(0),
                k_item.unsqueeze(0),
                v_item.unsqueeze(0),
                attn_bias=attn_bias,
                p=dropout_p,
                scale=softmax_scale,
            ).squeeze(0)
        else:
            attn_kwargs = {
                "attn_mask": None,
                "dropout_p": dropout_p,
                "is_causal": causal,
            }
            if softmax_scale is not None:
                attn_kwargs["scale"] = softmax_scale
            out = _worldfoundry_scaled_dot_product_attention(
                q_item.transpose(0, 1).unsqueeze(0),
                k_item.transpose(0, 1).unsqueeze(0),
                v_item.transpose(0, 1).unsqueeze(0),
                **attn_kwargs,
            ).squeeze(0).transpose(0, 1)

        outputs.append(out)
        q_start += q_len
        k_start += k_len

    if not outputs:
        return q.new_empty((batch, query_length, q.shape[1], v.shape[-1]))
    return torch.stack(outputs, dim=0)
