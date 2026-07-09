"""WorldFoundry facade for CrossLID diversity metric."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_crosslid() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _crosslid_fn() -> Any:
    _ensure_crosslid()
    from crosslid_core import compute_crosslid

    return compute_crosslid


def compute_crosslid(
    reference_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    k: int = 100,
    batch_size: int = 1000,
) -> float:
    """Compute Cross Local Intrinsic Dimensionality (lower is more diverse)."""
    return float(_crosslid_fn()(generated_features, reference_features, k=k, batch_size=batch_size))


__all__ = ["compute_crosslid", "package_root"]
