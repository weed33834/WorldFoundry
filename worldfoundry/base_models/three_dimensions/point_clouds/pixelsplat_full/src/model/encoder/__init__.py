"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> model -> encoder -> __init__.py functionality."""

from .encoder import Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg

ENCODERS = {
    "epipolar": (EncoderEpipolar, None),
}

EncoderCfg = EncoderEpipolarCfg


def get_encoder(cfg: EncoderCfg):
    """Get encoder.

    Args:
        cfg: The cfg.
    """
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder, None
