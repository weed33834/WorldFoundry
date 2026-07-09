from __future__ import annotations

from worldfoundry.base_models.diffusion_model.video.vchitect.worldfoundry_runtime import Vchitect

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class Vchitect2T2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for vchitect_2_t2v."""

    MODEL_NAME = "vchitect_2_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = Vchitect
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/vchitect/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
