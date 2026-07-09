"""CMMD (CLIP Maximum Mean Discrepancy) metric for image generation evaluation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.cmmd import wrapper as _wrapper_module

    return _wrapper_module


def compute_cmmd(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_cmmd(*args, **kwargs)


def compute_cmmd_from_embeddings(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_cmmd_from_embeddings(*args, **kwargs)


__all__ = [
    "compute_cmmd",
    "compute_cmmd_from_embeddings",
    "package_root",
]
