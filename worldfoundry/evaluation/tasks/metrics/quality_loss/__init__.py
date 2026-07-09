"""Quality Loss metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .wrapper import (
    compute_quality_loss,
    compute_quality_loss_for_pair,
    compute_quality_loss_from_batch,
    compute_text_presence_probability,
)

METRIC_ID = "quality_loss"
ALIASES = ("quality-loss", "ql")
HIGHER_IS_BETTER = False
FAMILY = "scorer"
TAGS = ("scorer", "text_to_image", "clip", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Quality Loss (IEEE Access 2023.3348778): CLIPScore multiplied by the probability "
        "that generated images contain unintended text/characters. "
        "WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(clip_score: float, text_presence_probability: float) -> float:
    return compute_quality_loss(clip_score, text_presence_probability)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_quality_loss",
    "compute_quality_loss_for_pair",
    "compute_quality_loss_from_batch",
    "compute_text_presence_probability",
]
