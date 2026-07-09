"""
Provides a wrapper for the Minimax Image-to-Video (I2V) synthesis model,
defining its configuration and runtime details.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .minimax_runtime import Minimax


class MiniMaxI2VSynthesis(RuntimeVideoSynthesis):
    """
    Implements an independent synthesis wrapper for the Minimax Image-to-Video (I2V) model.

    This class inherits from `RuntimeVideoSynthesis` and configures specific details
    for the Minimax I2V model, including its name, generation type,
    associated runtime class, and default runtime configuration path.
    """

    # The unique name of the model.
    MODEL_NAME = "minimax_i2v"
    # The type of generation this model performs (e.g., image-to-video).
    GENERATION_TYPE = "i2v"
    # The concrete runtime class responsible for executing the model.
    RUNTIME_CLS = Minimax
    # Specifies the primary path key for model assets, or None if not applicable.
    PRIMARY_PATH_KEY = None
    # The path to the default YAML configuration file for this runtime.
    RUNTIME_CONFIG_PATH = "models/runtime/configs/minimax/runtime_defaults.yaml"
    # The key under which this model's configuration is expected in a larger config structure.
    RUNTIME_CONFIG_KEY = MODEL_NAME