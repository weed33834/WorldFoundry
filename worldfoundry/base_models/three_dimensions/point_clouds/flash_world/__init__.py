"""Module for base_models -> three_dimensions -> point_clouds -> flash_world -> __init__.py functionality."""

__all__ = [
    "AutoencoderKLWan",
    "WanTransformer3DModel",
    "WANDecoderPixelAligned3DGSReconstructionModel",
]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name == "AutoencoderKLWan":
        from .autoencoder_kl_wan import AutoencoderKLWan

        return AutoencoderKLWan
    if name == "WanTransformer3DModel":
        from .transformer_wan import WanTransformer3DModel

        return WanTransformer3DModel
    if name == "WANDecoderPixelAligned3DGSReconstructionModel":
        from .reconstruction_model import WANDecoderPixelAligned3DGSReconstructionModel

        return WANDecoderPixelAligned3DGSReconstructionModel
    raise AttributeError(name)
