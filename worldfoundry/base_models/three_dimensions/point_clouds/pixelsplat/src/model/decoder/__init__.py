"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> model -> decoder -> __init__.py functionality."""

from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    """Get decoder.

    Args:
        decoder_cfg: The decoder cfg.
        dataset_cfg: The dataset cfg.

    Returns:
        The return value.
    """
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
