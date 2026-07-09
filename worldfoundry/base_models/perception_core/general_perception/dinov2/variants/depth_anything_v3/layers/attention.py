# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

"""Module for base_models -> perception_core -> general_perception -> dinov2 -> variants -> depth_anything_v3 -> layers -> attention.py functionality."""

import logging

from torch import nn

from worldfoundry.core.attention import QKNormRopeSelfAttention

logger = logging.getLogger("dinov2")


class Attention(QKNormRopeSelfAttention):
    """Depth Anything V3 self-attention with optional Q/K norm and RoPE."""
