"""Canonical LVDM foundation utilities used by multiple visual-generation runtimes."""

from __future__ import annotations

import sys

sys.modules.setdefault("lvdm", sys.modules[__name__])

from .common import (
    autocast,
    checkpoint,
    default,
    exists,
    extract_into_tensor,
    gather_data,
    identity,
    init_,
    isimage,
    ismap,
    max_neg_value,
    mean_flat,
    noise_like,
    shape_to_str,
    uniq,
)
from .distributions import (
    AbstractDistribution,
    DiagonalGaussianDistribution,
    DiracDistribution,
    normal_kl,
)
from .ema import LitEma

__all__ = [
    "AbstractDistribution",
    "DiagonalGaussianDistribution",
    "DiracDistribution",
    "LitEma",
    "autocast",
    "checkpoint",
    "default",
    "exists",
    "extract_into_tensor",
    "gather_data",
    "identity",
    "init_",
    "isimage",
    "ismap",
    "max_neg_value",
    "mean_flat",
    "noise_like",
    "normal_kl",
    "shape_to_str",
    "uniq",
]
