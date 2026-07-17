"""Inference-only configuration registration for Gamma's distilled causal model."""

from typing import Any, List

import attrs

from worldfoundry.core.configuration import cosmos_config as config
from worldfoundry.core.configuration.hydra import import_all_modules_from_package
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.conditioner import register_conditioner
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.tokenizer import register_tokenizer
from worldfoundry.synthesis.visual_generation.gamma_world.configs.distilled.model import register_model
from worldfoundry.synthesis.visual_generation.gamma_world.configs.distilled.net import register_net


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "fsdp_mv"},
            {"net": "causal_cosmosv2_2b"},
            {"conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    value = Config(model=None)
    register_conditioner()
    register_net()
    register_tokenizer()
    register_model()
    import_all_modules_from_package(
        "worldfoundry.synthesis.visual_generation.gamma_world.configs.distilled.experiment",
        reload=True,
    )
    return value
