"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> wan_video_vace.py functionality."""

from ...models.wan_video_vace_core import VaceWanModelDictConverter as _Converter


def VaceWanModelDictConverter(state_dict):
    """Vacewanmodeldictconverter.

    Args:
        state_dict: The state dict.
    """
    return _Converter().from_civitai(state_dict)

__all__ = ["VaceWanModelDictConverter"]
