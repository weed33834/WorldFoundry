# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> utils -> __init__.py functionality."""

from .utils import (
    cat_keep_shapes,
    count_parameters,
    named_apply,
    named_replace,
    uncat_with_shapes,
)
