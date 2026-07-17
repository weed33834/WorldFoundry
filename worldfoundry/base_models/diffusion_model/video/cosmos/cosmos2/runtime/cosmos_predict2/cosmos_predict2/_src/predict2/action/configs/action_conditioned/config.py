"""Inference-only action-conditioned Predict2 configuration entry point."""

from typing import Any

import attrs
from cosmos_predict2._src.predict2.action.configs.action_conditioned.inference_defaults import (
    register_action_inference,
)

from worldfoundry.core.configuration import Config as InferenceConfig


@attrs.define(slots=False)
class Config(InferenceConfig):
    defaults: list[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "action_rectified_flow"},
            {"net": "action_chunk_2B"},
            {"conditioner": "action_conditioner"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    register_action_inference()
    config = Config(model=None)
    config.job.project = "cosmos_predict2_action_conditioned"
    config.job.group = "inference"
    config.job.name = "action_inference"
    return config
