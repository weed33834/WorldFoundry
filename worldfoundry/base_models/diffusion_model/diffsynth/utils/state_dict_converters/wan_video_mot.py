"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> wan_video_mot.py functionality."""

from ...models.wan_video_motion_controller import (
    WanMotionControllerModelDictConverter as _Converter,
)


def WanVideoMotStateDictConverter(state_dict):
    """Wanvideomotstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_civitai(state_dict)


__all__ = ["WanVideoMotStateDictConverter"]
