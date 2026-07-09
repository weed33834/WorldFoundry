"""Improved Precision and Recall (α-precision / β-recall) metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .wrapper import compute_improved_precision_recall, compute_realism_score

METRIC_ID = "improved_precision_recall"
ALIASES = (
    "ipr",
    "alpha-precision",
    "beta-recall",
    "alpha_precision",
    "beta_recall",
    "realism_score",
    "ipr-realism",
    "realism",
    "ipr_realism",
)
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Improved Precision and Recall (Kynkäänniemi et al., 2019) for generative models. "
        "Includes per-image realism score (``compute_realism_score``) from the same IPR codebase."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.improved_precision_recall import wrapper as _wrapper_module

    return _wrapper_module


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    return compute_improved_precision_recall(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_improved_precision_recall",
    "compute_realism_score",
    "package_root",
]


def package_root() -> Path:
    return Path(__file__).resolve().parent
