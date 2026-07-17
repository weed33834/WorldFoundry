"""Inference-only DreamX-World Wan variants."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AutoencoderKLWan3_8",
    "CausalWanModel",
    "Wan2_2Transformer3DModel",
    "WanDiffusionCameraWrapper",
    "WanT5EncoderModel",
    "WanTextEncoder",
    "WanVAEWrapper",
]


def __getattr__(name: str) -> Any:
    if name == "CausalWanModel":
        from .causal_camera_model import CausalWanModel

        return CausalWanModel
    if name == "Wan2_2Transformer3DModel":
        from .transformer import Wan2_2Transformer3DModel

        return Wan2_2Transformer3DModel
    if name == "WanT5EncoderModel":
        from .text_encoder import WanT5EncoderModel

        return WanT5EncoderModel
    if name == "AutoencoderKLWan3_8":
        from .vae import AutoencoderKLWan3_8

        return AutoencoderKLWan3_8
    if name in {"WanDiffusionCameraWrapper", "WanTextEncoder", "WanVAEWrapper"}:
        from .wrappers import WanDiffusionCameraWrapper, WanTextEncoder, WanVAEWrapper

        return {
            "WanDiffusionCameraWrapper": WanDiffusionCameraWrapper,
            "WanTextEncoder": WanTextEncoder,
            "WanVAEWrapper": WanVAEWrapper,
        }[name]
    raise AttributeError(name)
