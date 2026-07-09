"""Lazy torch-fidelity loader with vendored package on ``sys.path``."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_torch_fidelity() -> None:
    root = str(_VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def calculate_metrics() -> Any:
    ensure_torch_fidelity()
    from torch_fidelity.metrics import calculate_metrics as _calculate_metrics

    return _calculate_metrics


__all__ = ["calculate_metrics", "ensure_torch_fidelity", "vendor_root"]
