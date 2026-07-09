# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable model building blocks shared across dvlt models.

The package is organized as flat modules at the top level (DINOv2 backbone,
pose-encoding helpers, head activations) plus two sub-packages with multiple
files each: :mod:`layers` (transformer primitives) and :mod:`loss` (training
losses). All publicly used symbols are re-exported here so that callers can
write ``from dvlt.model_components import Block`` instead of digging into
submodule paths.
"""

from .head_activations import activate_head
from .layers.attention import Attention, set_attn_backend
from .layers.block import Block, DropPath, LayerScale, Mlp, NestedTensorBlock
from .layers.patch_embed import PatchEmbed
from .layers.rope import PositionGetter, RotaryPositionEmbedding2D
from worldfoundry.core.nn.layers import SwiGLUFFNFused

from .pose_encoding import (
    create_uv_grid,
    extri_intri_to_pose_enc,
    pose_enc_to_extri_intri,
)


__all__ = [
    # Heads / pose helpers
    "activate_head",
    "create_uv_grid",
    "extri_intri_to_pose_enc",
    "pose_enc_to_extri_intri",
    # Transformer primitives
    "Attention",
    "set_attn_backend",
    "Block",
    "NestedTensorBlock",
    "DropPath",
    "LayerScale",
    "Mlp",
    "PatchEmbed",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
    "SwiGLUFFNFused",
]
