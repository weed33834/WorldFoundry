"""Dedicated Hy-Embodied-0.5-VLA pipeline target."""

from __future__ import annotations

__all__ = ["HyEmbodiedVLAPipeline"]


def __getattr__(name: str):
    if name == "HyEmbodiedVLAPipeline":
        from .pipeline_hy_embodied_vla import HyEmbodiedVLAPipeline

        return HyEmbodiedVLAPipeline
    raise AttributeError(name)
