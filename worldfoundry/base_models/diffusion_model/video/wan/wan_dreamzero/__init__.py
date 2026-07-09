"""Canonical DreamZero Wan foundation fork.

DreamZero action-generation code lives under ``worldfoundry.synthesis``. The Wan
backbone, schedulers, encoders, and VAE fork used by that runtime live here so
foundation-model code is owned by ``worldfoundry.base_models``.
"""

from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
MODULES_ROOT = SOURCE_ROOT / "modules"


__all__ = [
    "MODULES_ROOT",
    "SOURCE_ROOT",
]
