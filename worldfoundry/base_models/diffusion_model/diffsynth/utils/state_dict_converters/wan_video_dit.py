"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> wan_video_dit.py functionality."""

from ...models.wan_video_dit import WanModelStateDictConverter as _Converter


def WanVideoDiTStateDictConverter(state_dict):
    """Wanvideoditstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_civitai(state_dict)


def WanVideoDiTFromDiffusers(state_dict):
    """Wanvideoditfromdiffusers.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_diffusers(state_dict)


__all__ = ["WanVideoDiTStateDictConverter", "WanVideoDiTFromDiffusers"]
