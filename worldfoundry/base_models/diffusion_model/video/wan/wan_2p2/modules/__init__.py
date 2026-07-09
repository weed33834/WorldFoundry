# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p2 -> modules -> __init__.py functionality."""

from importlib import import_module

__all__ = [
    'Wan2_1_VAE',
    'Wan2_2_VAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'flash_attention',
]

_EXPORTS = {
    "Wan2_1_VAE": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_1",
    "Wan2_2_VAE": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2",
    "WanModel": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.model",
    "T5Model": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
    "T5Encoder": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
    "T5Decoder": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
    "T5EncoderModel": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
    "HuggingfaceTokenizer": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.tokenizers",
    "flash_attention": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.attention",
}


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
