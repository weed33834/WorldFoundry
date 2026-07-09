"""DINO similarity metric for condition consistency evaluation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.dino_similarity import wrapper as _wrapper_module

    return _wrapper_module


def compute_dino_similarity(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_dino_similarity(*args, **kwargs)


__all__ = [
    "compute_dino_similarity",
    "package_root",
]
