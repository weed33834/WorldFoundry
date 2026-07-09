"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> evaluation -> evaluation_index_generator.py functionality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexEntry:
    """Index entry implementation."""
    context: tuple[int, ...]
    target: tuple[int, ...]
