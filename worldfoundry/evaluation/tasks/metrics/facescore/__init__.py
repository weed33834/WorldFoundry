"""FaceScore face quality metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.facescore import wrapper as _wrapper_module

    return _wrapper_module


def FaceScoreModel(*args: Any, **kwargs: Any) -> Any:
    return _wrapper().FaceScoreModel(*args, **kwargs)


def compute_facescore(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_facescore(*args, **kwargs)


__all__ = ["FaceScoreModel", "compute_facescore", "package_root"]
