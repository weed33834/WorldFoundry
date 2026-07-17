# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

"""Compatibility exports for the shared WorldFoundry 2D RoPE implementation."""

from worldfoundry.core.attention.rope_2d import PositionGetter, RotaryPositionEmbedding2D

__all__ = ["PositionGetter", "RotaryPositionEmbedding2D"]
