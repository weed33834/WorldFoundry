"""Module for base_models -> diffusion_model -> video -> wan -> variants -> echo_infinity -> wan -> modules -> attention.py functionality."""

import torch
try:
    import flash_attn_interface

    def is_hopper_gpu():
        """Is hopper gpu."""
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 9
        return False
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
import warnings
__all__ = ['flash_attention', 'attention']

def _scaled_dot_product_attention_fallback(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, dtype=torch.bfloat16):
    """Helper function to scaled dot product attention fallback.

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
        dtype: The dtype.
    """
    if q_lens is not None or k_lens is not None:
        warnings.warn('Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.')
    if q_scale is not None:
        q = q * q_scale
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """Flash attention.

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
        version: The version.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        return _scaled_dot_product_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    b, lq, lk, out_dtype = (q.size(0), q.size(1), k.size(1), q.dtype)

    def half(x):
        """Half.

        Args:
            x: The x.
        """
        return x if x.dtype in half_dtypes else x.to(dtype)
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    if version is not None and version == 3 and (not FLASH_ATTN_3_AVAILABLE):
        warnings.warn('Flash attention 3 is not available, use flash attention 2 instead.')
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), max_seqlen_q=lq, max_seqlen_k=lk, softmax_scale=softmax_scale, causal=causal, deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, window_size=window_size, deterministic=deterministic).unflatten(0, (b, lq))
    return x.type(out_dtype)

def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, fa_version=None):
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
        return flash_attention(q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype, version=fa_version)
    else:
        return _scaled_dot_product_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
