"""Inference-only configuration registration for Gamma causal models."""

from typing import Any, List

import attrs

from worldfoundry.core.configuration import cosmos_config as config
from worldfoundry.core.configuration.hydra import import_all_modules_from_package
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.conditioner import register_conditioner
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.model import register_model
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.net import register_net
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.tokenizer import register_tokenizer


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "fsdp_mv"},
            {"net": None},
            {"conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    value = Config(model=None)
    register_model()
    register_net()
    register_conditioner()
    register_tokenizer()
    import_all_modules_from_package(
        "worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.experiment",
        reload=True,
    )
    return value
