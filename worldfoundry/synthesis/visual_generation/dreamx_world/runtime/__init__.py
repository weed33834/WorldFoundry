"""DreamX-World synthesis pipeline."""

from __future__ import annotations

from typing import Any

__all__ = ["Wan2_2_CameraPipeline"]


def __getattr__(name: str) -> Any:
    if name == "Wan2_2_CameraPipeline":
        from .pipeline import Wan2_2_CameraPipeline

        return Wan2_2_CameraPipeline
    raise AttributeError(name)
