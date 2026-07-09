from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .gen3_runtime import Gen3


class Gen3I2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for gen_3_i2v."""

    MODEL_NAME = "gen_3_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = Gen3
    PRIMARY_PATH_KEY = None
    RUNTIME_CONFIG_PATH = "models/runtime/configs/runway/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
