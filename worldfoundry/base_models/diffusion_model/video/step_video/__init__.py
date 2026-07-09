"""Module for base_models -> diffusion_model -> video -> step_video -> __init__.py functionality."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """Runtime root.

    Returns:
        The return value.
    """
    return Path(__file__).resolve().parent / 'step_video_runtime'


__all__ = ["runtime_root"]
