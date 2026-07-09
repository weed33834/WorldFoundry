"""Vendi Score diversity metric for generative model evaluation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.vendi_score import wrapper as _wrapper_module

    return _wrapper_module


def compute_vendi_score(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_vendi_score(*args, **kwargs)


def compute_vendi_score_from_features(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_vendi_score_from_features(*args, **kwargs)


__all__ = [
    "compute_vendi_score",
    "compute_vendi_score_from_features",
    "package_root",
]
