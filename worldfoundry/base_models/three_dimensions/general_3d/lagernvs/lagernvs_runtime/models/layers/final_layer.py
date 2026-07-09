# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module for base_models -> three_dimensions -> general_3d -> lagernvs -> lagernvs_runtime -> models -> layers -> final_layer.py functionality."""

import torch.nn as nn


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
    ):
        """Init.

        Args:
            hidden_size: The hidden size.
            patch_size: The patch size.
            out_channels: The out channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, bias=False)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=False
        )

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        x = self.norm_final(x)
        x = self.linear(x)
        return x
