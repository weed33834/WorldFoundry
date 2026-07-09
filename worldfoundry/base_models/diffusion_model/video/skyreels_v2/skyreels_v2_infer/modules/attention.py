# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> skyreels_v2 -> skyreels_v2_infer -> modules -> attention.py functionality."""

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

__all__ = [
    "flash_attention",
    "attention",
]


def _scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """Helper function to scaled dot product attention.

    Args:
        q: The q.
        k: The k.
        v: The v.
        dropout_p: The dropout p.
        softmax_scale: The softmax scale.
        q_scale: The q scale.
        causal: The causal.
        dtype: The dtype.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    q = q.to(dtype if q.dtype not in half_dtypes else q.dtype)
    k = k.to(dtype if k.dtype not in half_dtypes else k.dtype)
    v = v.to(dtype if v.dtype not in half_dtypes else v.dtype)
    if q_scale is not None:
        q = q * q_scale
    if q.shape[2] != k.shape[2]:
        if q.shape[2] % k.shape[2] != 0:
            raise ValueError(f"q heads ({q.shape[2]}) must be divisible by k heads ({k.shape[2]}).")
        repeat = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2).to(q.dtype)
    v = v.transpose(1, 2).to(q.dtype)
    kwargs = {
        "attn_mask": None,
        "is_causal": causal,
        "dropout_p": dropout_p,
    }
    if softmax_scale is not None:
        kwargs["scale"] = softmax_scale
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
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
    assert q.device.type == "cuda" and q.size(-1) <= 256
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        if window_size != (-1, -1):
            warnings.warn("Sliding-window attention is disabled for the torch SDPA fallback.")
        return _scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        """Half.

        Args:
            x: The x.
        """
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query

    q = half(q.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)

    # preprocess key, value

    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))
    k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")

    torch.cuda.nvtx.range_push(f"{list(q.shape)}-{list(k.shape)}-{list(v.shape)}-{q.dtype}-{k.dtype}-{v.dtype}")
    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    torch.cuda.nvtx.range_pop()

    # output
    return x


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
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
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
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
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        if window_size != (-1, -1):
            warnings.warn("Sliding-window attention is disabled for the torch SDPA fallback.")
        return _scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
