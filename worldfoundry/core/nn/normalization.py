"""Stateless normalization helpers for tensor-like arrays."""

from __future__ import annotations

from typing import Any


def rms_norm(value: Any, weight: Any = None, *, eps: float = 1e-6) -> Any:
    """Apply RMS normalization over the last dimension."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            squared_mean = (value * value).mean(dim=-1, keepdim=True)
            normalized = value * torch.rsqrt(squared_mean + float(eps))
            return normalized if weight is None else normalized * weight
    except ImportError:
        pass

    squared_mean = (value * value).mean(axis=-1, keepdims=True)
    normalized = value / ((squared_mean + float(eps)) ** 0.5)
    return normalized if weight is None else normalized * weight


def layer_scale(value: Any, scale: Any, *, bias: Any = None) -> Any:
    """Apply a stateless elementwise layer-scale transform."""

    output = value * scale
    return output if bias is None else output + bias


__all__ = ["layer_scale", "rms_norm"]
