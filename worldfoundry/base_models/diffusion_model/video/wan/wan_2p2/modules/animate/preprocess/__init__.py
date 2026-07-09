# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p2 -> modules -> animate -> preprocess -> __init__.py functionality."""

__all__ = ["ProcessPipeline", "SAM2VideoPredictor"]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name == "ProcessPipeline":
        from .process_pipepline import ProcessPipeline

        return ProcessPipeline
    if name == "SAM2VideoPredictor":
        from .video_predictor import SAM2VideoPredictor

        return SAM2VideoPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
