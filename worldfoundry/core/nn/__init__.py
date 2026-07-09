"""Small reusable neural-network tensor helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AttentionBackendInfo": "worldfoundry.core.attention.native",
    "DropPath": "worldfoundry.core.nn.layers",
    "FlowMatchScheduler": "worldfoundry.core.nn.diffusion_schedulers",
    "LayerNorm2d": "worldfoundry.core.nn.layers",
    "LayerScale": "worldfoundry.core.nn.layers",
    "SamHeadMLP": "worldfoundry.core.nn.layers",
    "SamMLPBlock": "worldfoundry.core.nn.layers",
    "SchedulerInterface": "worldfoundry.core.nn.diffusion_schedulers",
    "Mlp": "worldfoundry.core.nn.layers",
    "PositionEmbeddingRandom": "worldfoundry.core.nn.layers",
    "PatchGridSpec": "worldfoundry.core.nn.patching",
    "PatchEmbed": "worldfoundry.core.nn.layers",
    "PatchEmbed_Mlp": "worldfoundry.core.nn.layers",
    "Permute": "worldfoundry.core.nn.layers",
    "PixelUnshuffle": "worldfoundry.core.nn.layers",
    "PreNormTransformerBlock": "worldfoundry.core.nn.vit_block",
    "QKVSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "RopePreNormTransformerBlock": "worldfoundry.core.nn.vit_block",
    "SwiGLU": "worldfoundry.core.nn.layers",
    "SwiGLUFFN": "worldfoundry.core.nn.layers",
    "SwiGLUFFNFused": "worldfoundry.core.nn.layers",
    "XFORMERS_AVAILABLE": "worldfoundry.core.nn.layers",
    "XFORMERS_ENABLED": "worldfoundry.core.nn.layers",
    "TransformerShapeSpec": "worldfoundry.core.nn.transformer",
    "add_residual": "worldfoundry.core.nn.stochastic_depth",
    "apply_prenorm_transformer_residuals": "worldfoundry.core.nn.vit_block",
    "apply_rotary_embedding": "worldfoundry.core.attention.rope",
    "attention_backend_info": "worldfoundry.core.attention.native",
    "attention_head_dim": "worldfoundry.core.nn.transformer",
    "causal_attention_mask": "worldfoundry.core.nn.transformer",
    "drop_add_residual_stochastic_depth": "worldfoundry.core.nn.stochastic_depth",
    "drop_path": "worldfoundry.core.nn.layers",
    "get_branges_scales": "worldfoundry.core.nn.stochastic_depth",
    "layer_scale": "worldfoundry.core.nn.normalization",
    "merge_attention_heads": "worldfoundry.core.nn.transformer",
    "make_2tuple": "worldfoundry.core.nn.layers",
    "mlp_hidden_size": "worldfoundry.core.nn.transformer",
    "patchify_image": "worldfoundry.core.nn.patching",
    "rms_norm": "worldfoundry.core.nn.normalization",
    "rotary_frequencies": "worldfoundry.core.attention.rope",
    "rotate_half": "worldfoundry.core.attention.rope",
    "scaled_dot_product_attention": "worldfoundry.core.attention.native",
    "sinusoidal_embedding_1d": "worldfoundry.core.nn.transformer",
    "split_attention_heads": "worldfoundry.core.nn.transformer",
    "transformer_shape_spec": "worldfoundry.core.nn.transformer",
    "unpatchify_image": "worldfoundry.core.nn.patching",
    "zero_module": "worldfoundry.core.nn.layers",
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


__all__ = sorted(_EXPORT_MODULES)
