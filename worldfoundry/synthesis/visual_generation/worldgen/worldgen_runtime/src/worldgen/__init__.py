from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "WorldGen":
        from .worldgen import WorldGen

        return WorldGen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["WorldGen"]
