"""Semantic Shift Rate (SemSR) metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .wrapper import (
    compute_sem_shift,
    compute_semsr,
    compute_semsr_from_embeddings,
    compute_semsr_from_images,
    compute_semsr_from_similarities,
)

METRIC_ID = "semsr"
ALIASES = ("semantic-shift-rate", "semantic_shift_rate", "sem_sr")
HIGHER_IS_BETTER = True
FAMILY = "scorer"
TAGS = ("scorer", "text_to_image", "clip", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Semantic Shift Rate (SemSR, arXiv:2402.07562): normalized CLIP semantic shift "
        "between trigger/origin/target images. WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    return compute_semsr_from_similarities(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_sem_shift",
    "compute_semsr",
    "compute_semsr_from_embeddings",
    "compute_semsr_from_images",
    "compute_semsr_from_similarities",
]
