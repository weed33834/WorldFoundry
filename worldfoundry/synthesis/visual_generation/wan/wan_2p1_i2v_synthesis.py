from __future__ import annotations

from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis
from worldfoundry.base_models.diffusion_model.video.wan.wan_runtime_wrapper import Wan


class Wan2p1I2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for wan2.1_i2v."""

    MODEL_NAME = "wan2.1_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = Wan
    PRIMARY_PATH_KEY = "ckpt_dir"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/wan/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
