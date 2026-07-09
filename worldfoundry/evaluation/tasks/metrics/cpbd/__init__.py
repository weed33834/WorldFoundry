"""CPBD blur/sharpness metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.cpbd import wrapper as _wrapper_module

    return _wrapper_module


def compute_cpbd(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_cpbd(*args, **kwargs)


__all__ = ["compute_cpbd", "package_root"]
