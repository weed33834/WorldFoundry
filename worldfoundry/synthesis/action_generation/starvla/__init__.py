"""StarVLA public API with lazy profile loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["StarVLASynthesis"]


def __getattr__(name: str) -> Any:
    if name != "StarVLASynthesis":
        raise AttributeError(name)
    value = import_module(f"{__name__}.starvla_synthesis").StarVLASynthesis
    globals()[name] = value
    return value
