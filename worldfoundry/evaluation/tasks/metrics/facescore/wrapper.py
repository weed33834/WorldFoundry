"""WorldFoundry facade for FaceScore face quality metric."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_facescore() -> None:
    root = str(PACKAGE_ROOT / "facescore_pkg")
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _facescore_class() -> Any:
    _ensure_facescore()
    from FaceScore import FaceScore

    return FaceScore


def FaceScoreModel(
    model_name: str = "FaceScore",
    *,
    med_config: str | None = None,
    device: str = "cuda",
) -> Any:
    """Construct a FaceScore model (requires ImageReward + RetinaFace checkpoints)."""
    return _facescore_class()(model_name, med_config=med_config, device=device)


def compute_facescore(image_path: str | Path, *, model_name: str = "FaceScore", device: str = "cuda") -> float:
    """Score a single image path with FaceScore (higher is better)."""
    model = FaceScoreModel(model_name=model_name, device=device)
    score, _, _ = model.get_reward(str(image_path))
    return float(score)


__all__ = ["FaceScoreModel", "compute_facescore", "package_root"]
