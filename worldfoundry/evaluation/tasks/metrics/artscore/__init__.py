"""ArtScore artness evaluation metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _wrapper() -> Any:
    from worldfoundry.evaluation.tasks.metrics.artscore import wrapper as _wrapper_module

    return _wrapper_module


def load_artscore_model(*args: Any, **kwargs: Any) -> Any:
    return _wrapper().load_artscore_model(*args, **kwargs)


__all__ = ["load_artscore_model", "package_root"]
