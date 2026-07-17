# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only tensor-parallel linear and embedding adapters."""

import torch

from worldfoundry.core.distributed.megatron_compat import (
    ColumnParallelLinear as _ColumnParallelLinear,
)
from worldfoundry.core.distributed.megatron_compat import (
    RowParallelLinear as _RowParallelLinear,
)
from worldfoundry.core.distributed.megatron_compat import VocabUtility, parallel_state


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding sharded across the vocabulary dimension."""

    def __init__(self, num_embeddings: int, embedding_dim: int, precision: str = "bfloat16") -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            parallel_state.get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                embedding_dim,
                device=device,
                dtype=getattr(torch, precision),
            )
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tensor_model_parallel_size == 1:
            return self.weight[input_]

        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output = self.weight[masked_input]
        output[input_mask, :] = 0.0
        torch.distributed.all_reduce(output, group=parallel_state.get_tensor_model_parallel_group())
        return output


class ColumnParallelLinear(_ColumnParallelLinear):
    """Column-parallel linear returning only the output tensor."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(input_)
        return output


class RowParallelLinear(_RowParallelLinear):
    """Row-parallel linear returning only the output tensor."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(input_)
        return output


__all__ = ["ColumnParallelLinear", "RowParallelLinear", "VocabParallelEmbedding"]
