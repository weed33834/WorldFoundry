"""Variable-length attention primitives used by packed vision/language models."""

from __future__ import annotations

from typing import Any
import warnings

import torch

from worldfoundry.core.attention.native import scaled_dot_product_attention

try:
    import flash_attn_interface as _flash_attn_interface
except Exception:
    _flash_attn_interface = None

try:
    import flash_attn as _flash_attn
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:
    _flash_attn = None
    _flash_attn_varlen_func = None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    """Wan-compatible variable-length attention without runtime-package imports."""

    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError("dtype must be float16 or bfloat16.")

    batch, q_len, k_len, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(value: torch.Tensor) -> torch.Tensor:
        return value if value.dtype in half_dtypes else value.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.full((batch,), q_len, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([item[:length] for item, length in zip(q, q_lens)]))
        q_lens = q_lens.to(device=q.device, dtype=torch.int32, non_blocking=True)

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.full((batch,), k_len, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([item[:length] for item, length in zip(k, k_lens)]))
        v = half(torch.cat([item[:length] for item, length in zip(v, k_lens)]))
        k_lens = k_lens.to(device=k.device, dtype=torch.int32, non_blocking=True)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True)

    if version == 3 and _flash_attn_interface is None:
        warnings.warn("FlashAttention 3 is not available, falling back to FlashAttention 2 or PyTorch SDPA.")

    if (version is None or version == 3) and _flash_attn_interface is not None:
        output = _flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0]
    elif _flash_attn is not None:
        output = _flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
    else:
        if window_size != (-1, -1):
            warnings.warn("Sliding-window attention is ignored by the PyTorch SDPA fallback.")
        output = _varlen_attention_torch(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return output.unflatten(0, (batch, q_len)).to(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version: int | None = None,
    version: int | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for upstream Wan ``attention`` helpers."""

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
        version=fa_version if fa_version is not None else version,
    )


def varlen_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Run packed variable-length attention with FlashAttention when available."""

    if _flash_attn_varlen_func is not None:
        return _flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            **kwargs,
        )
    return _varlen_attention_torch(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _varlen_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    q_offsets = _offsets(cu_seqlens_q)
    k_offsets = _offsets(cu_seqlens_k)
    outputs: list[torch.Tensor] = []
    compute_in_fp32 = query.device.type == "cpu" and query.dtype in {torch.float16, torch.bfloat16}

    for index in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[index], q_offsets[index + 1]
        k_start, k_end = k_offsets[index], k_offsets[index + 1]
        query_states = query[q_start:q_end]
        key_states = key[k_start:k_end]
        value_states = value[k_start:k_end]

        key_states, value_states = _repeat_key_value_for_gqa(
            key_states,
            value_states,
            query_heads=int(query_states.shape[1]),
        )
        if compute_in_fp32:
            query_states = query_states.float()
            key_states = key_states.float()
            value_states = value_states.float()

        attn_mask = None
        if causal:
            attn_mask = _bottom_right_causal_mask(
                int(query_states.shape[0]),
                int(key_states.shape[0]),
                query_states.device,
            )[None, None, :, :]

        sdpa_kwargs: dict[str, Any] = {}
        if query_states.device.type == "cpu" or attn_mask is not None:
            sdpa_kwargs["backend"] = "math"

        attn_output = scaled_dot_product_attention(
            query_states.transpose(0, 1).unsqueeze(0).contiguous(),
            key_states.transpose(0, 1).unsqueeze(0).contiguous(),
            value_states.transpose(0, 1).unsqueeze(0).contiguous(),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=softmax_scale,
            **sdpa_kwargs,
        )
        outputs.append(attn_output.squeeze(0).transpose(0, 1).to(dtype=query.dtype))

    if not outputs:
        return query.new_empty((0, query.shape[1], query.shape[2]))
    return torch.cat(outputs, dim=0)


def _offsets(cu_seqlens: torch.Tensor) -> list[int]:
    return [int(item) for item in cu_seqlens.detach().cpu().tolist()]


def _bottom_right_causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    query_positions = torch.arange(q_len, device=device)[:, None]
    key_positions = torch.arange(k_len, device=device)[None, :]
    return key_positions <= query_positions + (k_len - q_len)


def _repeat_key_value_for_gqa(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_value_heads = int(key_states.shape[1])
    if key_value_heads == query_heads:
        return key_states, value_states
    if query_heads % key_value_heads:
        raise ValueError(f"Cannot expand {key_value_heads} KV heads to {query_heads} query heads.")
    repeats = query_heads // key_value_heads
    return (
        key_states.repeat_interleave(repeats, dim=1),
        value_states.repeat_interleave(repeats, dim=1),
    )


__all__ = ["attention", "flash_attention", "varlen_scaled_dot_product_attention"]
