"""BLIP2 feature-extractor loader backed by the in-tree LAVIS runtime."""

from __future__ import annotations

from typing import Any


def load_blip2_feature_extractor(*, device: str = "cpu", is_eval: bool = True) -> tuple[Any, Any, Any]:
    from worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models.lavis.models import (
        load_model_and_preprocess,
    )

    return load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=is_eval,
        device=device,
    )
