from __future__ import annotations

from worldfoundry.base_models.diffusion_model.video.videocrafter.worldfoundry_runtime import VideoCrafter

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class VideoCrafter1I2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for videocrafter1_i2v."""

    MODEL_NAME = "videocrafter1_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = VideoCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/videocrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
