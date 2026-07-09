# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p2 -> modules -> animate -> __init__.py functionality."""

from .model_animate import WanAnimateModel
from .clip import CLIPModel
__all__ = ['WanAnimateModel', 'CLIPModel']
