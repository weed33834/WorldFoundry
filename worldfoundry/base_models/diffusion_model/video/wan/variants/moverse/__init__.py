"""MoVerse-specific Wan inference models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CausalWanModel",
    "CausalMMDiTModel",
]


def __getattr__(name: str) -> Any:
    if name == "CausalWanModel":
        from .causal_model import CausalWanModel

        return CausalWanModel
    if name == "CausalMMDiTModel":
        from .causal_mmdit_model import CausalMMDiTModel

        return CausalMMDiTModel
    raise AttributeError(name)
