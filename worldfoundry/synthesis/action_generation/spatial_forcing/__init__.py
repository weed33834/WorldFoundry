"""Lazy public API for the Spatial-Forcing action runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["SpatialForcingSynthesis"]


def __getattr__(name: str) -> Any:
    if name != "SpatialForcingSynthesis":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.spatial_forcing_synthesis").SpatialForcingSynthesis
    globals()[name] = value
    return value
