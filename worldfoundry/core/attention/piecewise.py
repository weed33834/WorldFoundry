"""In-tree PISA piecewise sparse attention with exact portable fallback."""

from __future__ import annotations

import os

import torch

from worldfoundry.core.attention.native import native_sdpa_priority, scaled_dot_product_attention


def _pisa_device_eligible(device: torch.device | str) -> bool:
    parsed = torch.device(device)
    if parsed.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(parsed)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    # The vendored implementation uses Triton tensor descriptors/TMA. It is
    # selected for Hopper and data-center Blackwell only. Consumer Blackwell
    # has a materially different shared-memory budget and stays exact until a
    # separately measured launch policy exists.
    return (major, minor) == (9, 0) or major == 10


def piecewise_attention_available(device: torch.device | str | None = None) -> bool:
    """Return whether the in-tree TMA implementation is eligible on ``device``."""

    if not torch.cuda.is_available():
        return False
    parsed = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if not _pisa_device_eligible(parsed):
        return False
    try:
        from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="pisa-triton")
        import triton  # noqa: F401
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
        backends=native_sdpa_priority(q.device),
    )


def piecewise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    density: float = 0.1,
    block_size: int = 64,
    min_sequence_length: int | None = None,
    strict: bool = False,
) -> torch.Tensor:
    """Run PISA block-routed attention or an exact SDPA fallback.

    Inputs use ``[batch, heads, sequence, head_dim]``. ``density`` is the
    fraction of KV blocks computed exactly; the remaining blocks use centroid
    approximation. Because this changes model math, PISA is never selected by
    the generic dense-attention dispatcher without an explicit caller choice.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("piecewise_attention expects 4D [B, H, S, D] tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if k.shape[:-1] != v.shape[:-1] or q.shape[0] != k.shape[0]:
        raise ValueError("q, k and v must have compatible batch/sequence shapes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if not 0 < float(density) <= 1:
        raise ValueError("density must be in (0, 1]")
    if block_size not in {16, 32, 64, 128}:
        raise ValueError("block_size must be one of 16, 32, 64 or 128")
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        if strict:
            raise RuntimeError("the PISA TMA kernel does not support GQA/MQA")
        return _exact_attention(q, k, v, scale)

    configured_min = os.getenv("WORLDFOUNDRY_PISA_MIN_SEQUENCE")
    threshold = int(configured_min) if configured_min is not None else 1024
    if min_sequence_length is not None:
        threshold = max(int(min_sequence_length), 0)
    eligible = (
        density < 1.0
        and min(q.shape[-2], k.shape[-2]) >= threshold
        and q.dtype in {torch.float16, torch.bfloat16}
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and 0 < q.shape[-1] <= 256
        and 0 < k.shape[-1] <= 256
        and 0 < v.shape[-1] <= 256
        and piecewise_attention_available(q.device)
    )
    if not eligible:
        if strict and density < 1.0:
            raise RuntimeError("the in-tree PISA TMA kernel is not eligible for this workload")
        return _exact_attention(q, k, v, scale)

    try:
        from worldfoundry.core.attention.triton_piecewise_attention import piecewise_attention_tma

        return piecewise_attention_tma(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=scale,
            density=float(density),
            block_size=int(block_size),
        )
    except Exception as exc:
        message = str(exc).casefold()
        if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
            marker in message for marker in ("out of memory", "alloc_failed")
        ):
            raise
        from_triton = exc.__class__.__module__.startswith("triton")
        optional_failure = isinstance(exc, (ImportError, OSError)) or any(
            marker in message
            for marker in ("triton", "ptxas", "no kernel image", "invalid device function", "out of resource")
        ) or from_triton
        if strict or not optional_failure:
            raise
        return _exact_attention(q, k, v, scale)


__all__ = ["piecewise_attention", "piecewise_attention_available"]
