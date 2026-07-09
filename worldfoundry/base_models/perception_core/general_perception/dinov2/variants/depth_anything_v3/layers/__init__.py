# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from .attention import MemEffAttention
"""Module for base_models -> perception_core -> general_perception -> dinov2 -> variants -> depth_anything_v3 -> layers -> __init__.py functionality."""

from .block import Block
from .rope import PositionGetter, RotaryPositionEmbedding2D

__all__ = [
    "Block",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
]
