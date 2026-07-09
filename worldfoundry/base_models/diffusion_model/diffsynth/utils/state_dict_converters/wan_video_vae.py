"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> wan_video_vae.py functionality."""

from ...models.scope_wan_video_vae import WanVideoVAEStateDictConverter as _Converter


def WanVideoVAEStateDictConverter(state_dict):
    """Wanvideovaestatedictconverter.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_civitai(state_dict)

__all__ = ["WanVideoVAEStateDictConverter"]
