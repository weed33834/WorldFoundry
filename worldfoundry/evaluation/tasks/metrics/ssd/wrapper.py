"""WorldFoundry facade for Semantic Similarity Distance (SSD)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _ensure_ssd() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _ssd_fn() -> Any:
    _ensure_ssd()
    import ssd_core as mod

    return mod.ssd


def compute_ssd(
    real_outputs: np.ndarray,
    generated_outputs: np.ndarray,
    conditions: np.ndarray,
) -> float:
    """Compute SSD from paired condition/output embedding arrays (TensorFlow backend)."""
    import tensorflow as tf

    y_true = tf.constant(real_outputs, dtype=tf.float32)
    y_predict = tf.constant(generated_outputs, dtype=tf.float32)
    x_true = tf.constant(conditions, dtype=tf.float32)
    total, _, _, _ = _ssd_fn()(y_true, y_predict, x_true)
    return float(total.numpy())


__all__ = ["compute_ssd", "package_root"]
