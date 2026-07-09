"""Conditional Inception Score (CIS) metric."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .wrapper import compute_bcis, compute_cis, compute_cis_from_predictions, compute_wcis

METRIC_ID = "cis"
ALIASES = ("conditional-inception-score", "conditional_is", "bcis", "wcis")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "conditional_generation", "inception_score")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Conditional Inception Score (Benny et al., IJCV 2020): BCIS * WCIS from "
        "class-bucketed softmax predictions."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    return compute_cis(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_bcis",
    "compute_cis",
    "compute_cis_from_predictions",
    "compute_wcis",
]
