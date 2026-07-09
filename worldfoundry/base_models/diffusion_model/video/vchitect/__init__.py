"""Module for base_models -> diffusion_model -> video -> vchitect -> __init__.py functionality."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """Runtime root.

    Returns:
        The return value.
    """
    return Path(__file__).resolve().parent / 'vchitect_runtime'


from .worldfoundry_runtime import Vchitect


__all__ = ["Vchitect", "runtime_root"]
