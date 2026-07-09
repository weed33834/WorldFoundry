"""
This module serves as the main entry point for the VideoCrafter model package,
re-exporting various VideoCrafter models for easy access.

It includes implementations for image-to-video (I2V) and text-to-video (T2V)
synthesis using different versions of the VideoCrafter architecture.
"""
from __future__ import annotations

from .videocrafter1_i2v_synthesis import VideoCrafter1I2VSynthesis
from .videocrafter1_t2v_synthesis import VideoCrafter1T2VSynthesis
from .videocrafter2_t2v_synthesis import VideoCrafter2T2VSynthesis

__all__ = [
    "VideoCrafter1I2VSynthesis",
    "VideoCrafter1T2VSynthesis",
    "VideoCrafter2T2VSynthesis",
]