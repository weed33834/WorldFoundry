# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Module for base_models -> perception_core -> segment -> sam3 -> __init__.py functionality."""

from typing import Any

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_predictor"]


def __getattr__(name: str) -> Any:
    """Getattr.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    if name in __all__:
        from . import model_builder

        return getattr(model_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
