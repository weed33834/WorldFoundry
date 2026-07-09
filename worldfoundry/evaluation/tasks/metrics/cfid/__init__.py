"""Conditional FID (CFID) metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.cfid import wrapper as _wrapper_module

    return _wrapper_module


def compute_cfid(*args: Any, **kwargs: Any) -> float:
    return _wrapper().compute_cfid(*args, **kwargs)


__all__ = ["compute_cfid", "package_root"]
