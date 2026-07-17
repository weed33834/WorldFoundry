"""Encoder factory restricted to the MapAnything inference closure."""

from .base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)
from .dense_rep_encoder import DenseRepresentationEncoder
from .dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner
from .global_rep_encoder import GlobalRepresentationEncoder

ENCODER_CONFIGS = {
    "dense_rep_encoder": DenseRepresentationEncoder,
    "dinov2": DINOv2Encoder,
    "global_rep_encoder": GlobalRepresentationEncoder,
}


def encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """Construct an encoder supported by the in-tree inference bundle."""
    try:
        encoder_class = ENCODER_CONFIGS[encoder_str]
    except KeyError as exc:
        supported = ", ".join(sorted(ENCODER_CONFIGS))
        raise ValueError(f"Unsupported inference encoder {encoder_str!r}; expected one of: {supported}") from exc
    return encoder_class(**kwargs)


__all__ = [
    "DINOv2Encoder",
    "DINOv2IntermediateFeatureReturner",
    "DenseRepresentationEncoder",
    "EncoderGlobalRepInput",
    "EncoderInput",
    "GlobalRepresentationEncoder",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "ViTEncoderInput",
    "ViTEncoderNonImageInput",
    "ViTEncoderOutput",
    "encoder_factory",
]
