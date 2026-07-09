"""Module for base_models -> diffusion_model -> video -> hunyuan_video -> modules -> __init__.py functionality."""

from .loader import load_model_from_registry
from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG


def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    return load_model_from_registry(
        args,
        in_channels,
        out_channels,
        factor_kwargs,
        HYVideoDiffusionTransformer,
        HUNYUAN_VIDEO_CONFIG,
    )
