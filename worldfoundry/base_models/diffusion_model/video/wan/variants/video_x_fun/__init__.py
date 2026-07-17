"""VideoX-Fun Wan inference models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AutoencoderKLWan",
    "AutoencoderKLWan3_8",
    "AutoTokenizer",
    "CLIPModel",
    "Wan2_2Transformer3DModel",
    "WanT5EncoderModel",
    "WanTransformer3DModel",
]


def __getattr__(name: str) -> Any:
    if name == "AutoTokenizer":
        from transformers import AutoTokenizer

        return AutoTokenizer
    if name == "CLIPModel":
        from .image_encoder import CLIPModel

        return CLIPModel
    if name == "WanT5EncoderModel":
        from worldfoundry.base_models.diffusion_model.video.wan.variants.dreamx_world.text_encoder import (
            WanT5EncoderModel,
        )

        return WanT5EncoderModel
    if name == "AutoencoderKLWan3_8":
        from worldfoundry.base_models.diffusion_model.video.wan.variants.dreamx_world.vae import (
            AutoencoderKLWan3_8,
        )

        return AutoencoderKLWan3_8
    if name == "AutoencoderKLWan":
        from .vae import AutoencoderKLWan

        return AutoencoderKLWan
    if name in {"Wan2_2Transformer3DModel", "WanTransformer3DModel"}:
        from .transformer import Wan2_2Transformer3DModel, WanTransformer3DModel

        return {
            "Wan2_2Transformer3DModel": Wan2_2Transformer3DModel,
            "WanTransformer3DModel": WanTransformer3DModel,
        }[name]
    raise AttributeError(name)
