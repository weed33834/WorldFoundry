"""Spatia's VACE head built from the shared DreamZero Wan backbone."""

from worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_vace_core import (
    build_vace_wan_classes,
)
from worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero.modules.wan_video_dit import (
    DiTBlock,
)
from worldfoundry.core.model_loading import hash_state_dict_keys

VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter = build_vace_wan_classes(
    DiTBlock,
    hash_state_dict_keys,
    module_name=__name__,
)

__all__ = ["VaceWanAttentionBlock", "VaceWanModel", "VaceWanModelDictConverter"]
