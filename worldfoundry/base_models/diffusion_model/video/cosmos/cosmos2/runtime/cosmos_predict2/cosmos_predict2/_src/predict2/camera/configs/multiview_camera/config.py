"""Inference-only camera-conditioned Predict2 configuration entry point."""

from typing import Any

import attrs
from cosmos_predict2._src.predict2.camera.configs.multiview_camera.inference_defaults import (
    register_camera_inference,
)

from worldfoundry.core.configuration import Config as InferenceConfig


@attrs.define(slots=False)
class Config(InferenceConfig):
    defaults: list[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "camera_frameinit_model"},
            {"net": "camera_net"},
            {"conditioner": "camera_frameinit_conditioner"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    register_camera_inference()
    config = Config(model=None)
    config.job.project = "cosmos_predict2_camera_conditioned"
    config.job.group = "inference"
    config.job.name = "camera_inference"
    return config
