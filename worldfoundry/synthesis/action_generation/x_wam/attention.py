"""WorldFoundry attention adapter for the X-WAM joint transformer."""

from __future__ import annotations

from typing import Any


def attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    causal: bool = False,
    dtype: Any = None,
    **_: Any,
) -> Any:
    """Run X-WAM's ``BSHD`` attention through the shared core dispatcher."""

    from worldfoundry.core.attention import scaled_dot_product_attention

    # Preserve the model/runtime-selected dtype. Forcing bf16 here breaks
    # Turing-class GPUs and changes CPU numerical behavior; the shared core
    # dispatcher already selects the best kernel for the active device.
    selected_dtype = q.dtype if dtype is None else dtype
    query = q.transpose(1, 2).to(selected_dtype)
    key = k.transpose(1, 2).to(selected_dtype)
    value = v.transpose(1, 2).to(selected_dtype)
    result = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
    return result.transpose(1, 2).contiguous()


__all__ = ["attention"]
