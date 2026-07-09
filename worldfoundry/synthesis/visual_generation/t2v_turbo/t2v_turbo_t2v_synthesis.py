from __future__ import annotations

from worldfoundry.base_models.diffusion_model.video.t2v_turbo.t2v_turbo_runtime import T2VTurbo

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class T2VTurboT2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for t2v_turbo_t2v."""

    MODEL_NAME = "t2v_turbo_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = T2VTurbo
    PRIMARY_PATH_KEY = "model_ckpt"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/t2v_turbo/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
