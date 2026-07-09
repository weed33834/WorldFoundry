# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Module for base_models -> perception_core -> general_perception -> dinov2 -> hub -> utils.py functionality."""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    """Helper function to make dinov2 model name.

    Args:
        arch_name: The arch name.
        patch_size: The patch size.
        num_register_tokens: The num register tokens.

    Returns:
        The return value.
    """
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


class CenterPadding(nn.Module):
    """Center padding implementation."""
    def __init__(self, multiple):
        """Init.

        Args:
            multiple: The multiple.
        """
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        """Helper function to get pad.

        Args:
            size: The size.
        """
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
