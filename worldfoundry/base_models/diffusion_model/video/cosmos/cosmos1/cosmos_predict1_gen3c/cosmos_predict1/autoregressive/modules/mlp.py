# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> autoregressive -> modules -> mlp.py functionality."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from worldfoundry.core.distributed.megatron_compat import parallel_state


def compute_llama3_ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: float) -> int:
    """
    Computes the feedforward network dimensionality.

    Args:
        dim (int): The embedding dimensionality.
        multiple_of (int): The multiple to round up the hidden dimensionality.
        ffn_dim_multiplier (float): The multiplier for the hidden dimensionality.

    Returns:
        The feedforward network dimensionality.
    """
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)  # custom dim factor
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    # Round up hidden dimensionality to the nearest multiple
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class MLP(nn.Module):
    """Mlp implementation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        tensor_parallel_size: int = 1,
    ):
        """
        Initializes the multilayer perceptron (MLP) module.

        Args:
            dim: The input and output dimensionality.
            hidden_dim: The dimensionality of the hidden layer.
        """
        super().__init__()
        self.tp_size = tensor_parallel_size
        self.w1 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)
        self.w2 = nn.Linear(hidden_dim // self.tp_size, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x: The input tensor of shape (batch_size, dim).

        Returns:
            The output tensor of shape (batch_size, dim).
        """
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.tp_size > 1:
            output = all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output
