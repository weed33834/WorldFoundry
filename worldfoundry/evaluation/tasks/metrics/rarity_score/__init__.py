"""Rarity Score metric for evaluating uncommonness of synthesized images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.rarity_score import wrapper as _wrapper_module

    return _wrapper_module


def compute_rarity_scores(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    return _wrapper().compute_rarity_scores(*args, **kwargs)


def compute_mean_rarity_score(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_mean_rarity_score(*args, **kwargs)


__all__ = [
    "compute_mean_rarity_score",
    "compute_rarity_scores",
    "package_root",
]
