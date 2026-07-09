"""
Module defining the Wan2p1T2VSynthesis class, an independent wrapper for the wan2.1_t2v video synthesis model.

This module provides a concrete implementation of RuntimeVideoSynthesis for the Wan 2.1 text-to-video model,
configuring it with specific paths and runtime settings.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis
from worldfoundry.base_models.diffusion_model.video.wan.wan_runtime_wrapper import Wan


class Wan2p1T2VSynthesis(RuntimeVideoSynthesis):
    """
    Independent synthesis wrapper for the wan2.1_t2v model.

    This class extends RuntimeVideoSynthesis to provide a specific configuration
    and runtime environment for the Wan 2.1 text-to-video model. It defines
    model-specific parameters such as its name, generation type, runtime class,
    and configuration paths.
    """

    MODEL_NAME = "wan2.1_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = Wan
    PRIMARY_PATH_KEY = "ckpt_dir"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/wan/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME