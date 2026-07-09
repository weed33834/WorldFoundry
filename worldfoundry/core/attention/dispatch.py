"""Dispatching attention wrapper for model code that uses flexible QKV layouts."""

import torch
from einops import rearrange
from worldfoundry.core.attention.backends import (
    attention_backend_capability,
    gpu_supports_flash_attention,
    probe_attention_backends,
    resolve_attention_backend,
)
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention


def initialize_attention_priority():
    return resolve_attention_backend()


ATTENTION_IMPLEMENTATION = initialize_attention_priority()
_CAPABILITIES = probe_attention_backends()
FLASH_ATTN_3_AVAILABLE = _CAPABILITIES["flash_attention_3"].available
FLASH_ATTN_2_AVAILABLE = _CAPABILITIES["flash_attention_2"].available
SAGE_ATTN_AVAILABLE = _CAPABILITIES["sage_attention"].available
XFORMERS_AVAILABLE = _CAPABILITIES["xformers"].available


def _gpu_supports_flash_attention():
    return gpu_supports_flash_attention()


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    try:
        out = _worldfoundry_scaled_dot_product_attention(q, k, v, attn_mask, scale=scale)
    except RuntimeError:
        scale = scale or (1.0 / (q.shape[-1] ** 0.5))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    import flash_attn_interface

    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    import flash_attn

    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    from sageattention import sageattn

    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    import xformers.ops as xops

    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, compatibility_mode=False):
    if compatibility_mode or (attn_mask is not None):
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale)
    try:
        selected = resolve_attention_backend(ATTENTION_IMPLEMENTATION)
        if selected == "flash_attention_3" and attention_backend_capability("flash_attention_3").usable:
            return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif selected == "flash_attention_2" and attention_backend_capability("flash_attention_2").usable:
            return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif selected == "sage_attention" and attention_backend_capability("sage_attention").usable:
            return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif selected == "xformers" and attention_backend_capability("xformers").usable:
            return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    except RuntimeError:
        pass
    return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)


def packed_sequence_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, scale=None):
    return attention_forward(
        q,
        k,
        v,
        q_pattern="b s (n d)",
        k_pattern="b s (n d)",
        v_pattern="b s (n d)",
        out_pattern="b s (n d)",
        dims={"n": num_heads},
        scale=scale,
        compatibility_mode=compatibility_mode,
    )


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "attention_forward",
    "flash_attention_2",
    "flash_attention_3",
    "initialize_attention_priority",
    "packed_sequence_attention",
    "rearrange_out",
    "rearrange_qkv",
    "sage_attention",
    "torch_sdpa",
    "xformers_attention",
]
