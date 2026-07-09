"""Optimized attention primitives and KV cache for streaming inference."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AttentionKernelCapability": "worldfoundry.core.attention.backends",
    "AttentionBackendInfo": "worldfoundry.core.attention.native",
    "BlockKVCache": "worldfoundry.core.attention.kvcache",
    "ContextParallelAttention": "worldfoundry.core.attention.cp",
    "KVCacheRelativeRotaryPositionEmbedding3D": "worldfoundry.core.attention.rope",
    "ModelMetaArgs": "worldfoundry.core.attention.packed_sequence",
    "NativeAttention": "worldfoundry.core.attention.native",
    "PackedCoreAttnParams": "worldfoundry.core.attention.packed_sequence",
    "PackedCrossAttnParams": "worldfoundry.core.attention.packed_sequence",
    "PositionGetter": "worldfoundry.core.attention.rope_2d",
    "RotaryPositionEmbedding3D": "worldfoundry.core.attention.rope",
    "RotaryPositionEmbedding2D": "worldfoundry.core.attention.rope_2d",
    "attention_backend_capability": "worldfoundry.core.attention.backends",
    "attention_backend_from_env": "worldfoundry.core.attention.backends",
    "attention_forward": "worldfoundry.core.attention.dispatch",
    "flash_attention": "worldfoundry.core.attention.varlen",
    "apply_nd_rotary_embedding": "worldfoundry.core.attention.rope_nd",
    "apply_rope_freqs": "worldfoundry.core.attention.rope",
    "apply_rotary_embedding": "worldfoundry.core.attention.rope",
    "attention_backend_report": "worldfoundry.core.attention.backends",
    "attention_backend_info": "worldfoundry.core.attention.native",
    "attention_backend_context": "worldfoundry.core.attention.native",
    "get_1d_rotary_pos_embed": "worldfoundry.core.attention.rope_nd",
    "get_meshgrid_nd": "worldfoundry.core.attention.rope_nd",
    "get_nd_rotary_pos_embed": "worldfoundry.core.attention.rope_nd",
    "gpu_supports_flash_attention": "worldfoundry.core.attention.backends",
    "normalize_attention_backend": "worldfoundry.core.attention.backends",
    "packed_sequence_attention": "worldfoundry.core.attention.dispatch",
    "attention": "worldfoundry.core.attention.varlen",
    "probe_attention_backends": "worldfoundry.core.attention.backends",
    "reshape_rotary_for_broadcast": "worldfoundry.core.attention.rope_nd",
    "rotary_frequencies": "worldfoundry.core.attention.rope",
    "rotate_half": "worldfoundry.core.attention.rope",
    "QKVSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "CSOHelper": "worldfoundry.core.attention.context_parallel_runtime",
    "UlyssesScheduler": "worldfoundry.core.attention.context_parallel_runtime",
    "cp_post_process": "worldfoundry.core.attention.context_parallel_runtime",
    "cp_pre_process": "worldfoundry.core.attention.context_parallel_runtime",
    "cso_communication": "worldfoundry.core.attention.context_parallel_runtime",
    "scaled_dot_product_attention": "worldfoundry.core.attention.native",
    "resolve_attention_backend": "worldfoundry.core.attention.backends",
    "varlen_scaled_dot_product_attention": "worldfoundry.core.attention.varlen",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = [
    "AttentionKernelCapability",
    "BlockKVCache",
    "ContextParallelAttention",
    "AttentionBackendInfo",
    "KVCacheRelativeRotaryPositionEmbedding3D",
    "ModelMetaArgs",
    "NativeAttention",
    "PackedCoreAttnParams",
    "PackedCrossAttnParams",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
    "RotaryPositionEmbedding3D",
    "attention_backend_capability",
    "attention_backend_from_env",
    "attention_backend_report",
    "attention_forward",
    "attention",
    "apply_nd_rotary_embedding",
    "apply_rope_freqs",
    "apply_rotary_embedding",
    "attention_backend_info",
    "attention_backend_context",
    "gpu_supports_flash_attention",
    "flash_attention",
    "get_1d_rotary_pos_embed",
    "get_meshgrid_nd",
    "get_nd_rotary_pos_embed",
    "normalize_attention_backend",
    "packed_sequence_attention",
    "probe_attention_backends",
    "QKVSelfAttention",
    "QKNormRopeSelfAttention",
    "CSOHelper",
    "UlyssesScheduler",
    "cp_post_process",
    "cp_pre_process",
    "cso_communication",
    "reshape_rotary_for_broadcast",
    "rotary_frequencies",
    "rotate_half",
    "scaled_dot_product_attention",
    "resolve_attention_backend",
    "varlen_scaled_dot_product_attention",
]
