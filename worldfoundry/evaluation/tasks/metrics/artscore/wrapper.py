"""WorldFoundry facade for ArtScore artness metric."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_artscore() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _get_resnet() -> Any:
    _ensure_artscore()
    from models import get_resnet

    return get_resnet


def load_artscore_model(checkpoint_path: str | Path, *, device: str = "cuda", **model_kwargs: Any) -> Any:
    """Load ArtScore ResNet checkpoint."""
    import torch

    class _Args:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    args = _Args(no_dense_layer=model_kwargs.get("no_dense_layer", False), **model_kwargs)
    model = _get_resnet()(args)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()
    return model


__all__ = ["load_artscore_model", "package_root"]
