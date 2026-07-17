"""MagicWorld-specific Wan inference models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CausalWanModel",
    "WanModel",
]


def __getattr__(name: str) -> Any:
    if name == "CausalWanModel":
        from .causal_model import CausalWanModel

        return CausalWanModel
    if name == "WanModel":
        from .model import WanModel

        return WanModel
    raise AttributeError(name)
