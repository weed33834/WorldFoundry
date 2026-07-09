"""WorldFoundry facade for Rarity Score."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_rarity_score() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _manifold_class() -> Any:
    _ensure_rarity_score()
    from rarity_score import MANIFOLD

    return MANIFOLD


def compute_rarity_scores(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    k: int = 3,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample rarity scores (higher means rarer)."""
    manifold = _manifold_class()(
        np.asarray(real_features),
        np.asarray(fake_features),
        device=device,
    )
    return manifold.rarity(k=k)


def compute_mean_rarity_score(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    k: int = 3,
    device: str = "cpu",
) -> float:
    """Compute mean rarity score over valid generated samples."""
    scores, score_ids = compute_rarity_scores(real_features, fake_features, k=k, device=device)
    if len(score_ids) == 0:
        return 0.0
    return float(np.mean(scores[score_ids]))


__all__ = [
    "compute_mean_rarity_score",
    "compute_rarity_scores",
    "package_root",
]
