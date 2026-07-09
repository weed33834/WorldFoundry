"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> wan_video_image_encoder.py functionality."""

from ...models.wan_video_image_encoder import WanImageEncoderStateDictConverter as _Converter


def WanImageEncoderStateDictConverter(state_dict):
    """Wanimageencoderstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_civitai(state_dict)

__all__ = ["WanImageEncoderStateDictConverter"]
