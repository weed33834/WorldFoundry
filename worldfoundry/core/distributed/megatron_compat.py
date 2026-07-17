"""Compatibility wrapper for optional Megatron model-parallel state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from megatron.core import ModelParallelConfig, mpu, parallel_state
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.tensor_parallel.utils import VocabUtility
except Exception:  # pragma: no cover - Megatron is optional for many runtimes.

    @dataclass
    class ModelParallelConfig:
        pipeline_model_parallel_size: int = 1
        tensor_model_parallel_size: int = 1
        context_parallel_size: int = 1
        sequence_parallel: bool = False

    class _NoParallelState:
        _initialized = False
        _context_parallel_group = None
        _context_parallel_size = 1
        _context_parallel_rank = 0
        _data_parallel_group = None
        _data_parallel_rank = 0
        _data_parallel_size = 1

        @classmethod
        def is_initialized(cls):
            return cls._initialized

        @classmethod
        def initialize_model_parallel(cls, *args, **kwargs):
            import torch.distributed as dist

            context_parallel_size = int(kwargs.get("context_parallel_size", 1) or 1)
            if not dist.is_available() or not dist.is_initialized():
                cls._initialized = True
                cls._context_parallel_size = 1
                cls._context_parallel_rank = 0
                return None

            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if context_parallel_size < 1:
                context_parallel_size = 1
            if world_size % context_parallel_size != 0:
                raise RuntimeError(
                    f"world_size {world_size} must be divisible by context_parallel_size {context_parallel_size}"
                )

            if context_parallel_size == world_size:
                cls._context_parallel_group = dist.group.WORLD
                cls._context_parallel_rank = rank
            else:
                for start in range(0, world_size, context_parallel_size):
                    ranks = list(range(start, start + context_parallel_size))
                    group = dist.new_group(ranks=ranks)
                    if rank in ranks:
                        cls._context_parallel_group = group
                        cls._context_parallel_rank = ranks.index(rank)

            data_parallel_size = world_size // context_parallel_size
            if data_parallel_size == 1:
                cls._data_parallel_group = None
                cls._data_parallel_rank = 0
            else:
                context_offset = rank % context_parallel_size
                for offset in range(context_parallel_size):
                    ranks = list(range(offset, world_size, context_parallel_size))
                    group = dist.new_group(ranks=ranks)
                    if offset == context_offset:
                        cls._data_parallel_group = group
                        cls._data_parallel_rank = ranks.index(rank)

            cls._context_parallel_size = context_parallel_size
            cls._data_parallel_size = data_parallel_size
            cls._initialized = True
            return None

        @classmethod
        def destroy_model_parallel(cls):
            cls._initialized = False
            cls._context_parallel_group = None
            cls._context_parallel_size = 1
            cls._context_parallel_rank = 0
            cls._data_parallel_group = None
            cls._data_parallel_rank = 0
            cls._data_parallel_size = 1
            return None

        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 1

        @staticmethod
        def get_tensor_model_parallel_rank():
            return 0

        @staticmethod
        def get_tensor_model_parallel_group():
            return None

        @classmethod
        def get_context_parallel_world_size(cls):
            return cls._context_parallel_size

        @classmethod
        def get_context_parallel_rank(cls):
            return cls._context_parallel_rank

        @classmethod
        def get_context_parallel_group(cls):
            return cls._context_parallel_group

        @classmethod
        def get_data_parallel_group(cls, with_context_parallel: bool = False):
            del with_context_parallel
            return cls._data_parallel_group

        @classmethod
        def get_data_parallel_rank(cls, with_context_parallel: bool = False):
            del with_context_parallel
            return cls._data_parallel_rank

        @classmethod
        def get_data_parallel_world_size(cls, with_context_parallel: bool = False):
            del with_context_parallel
            return cls._data_parallel_size

        @staticmethod
        def get_pipeline_model_parallel_rank():
            return 0

    parallel_state = _NoParallelState()
    mpu = parallel_state

    class VocabUtility:
        """Single-process vocabulary partition helper."""

        @staticmethod
        def vocab_range_from_global_vocab_size(global_size: int, rank: int, world_size: int) -> tuple[int, int]:
            if global_size % world_size != 0:
                raise ValueError(
                    f"vocabulary size {global_size} must be divisible by tensor parallel size {world_size}"
                )
            partition_size = global_size // world_size
            return rank * partition_size, (rank + 1) * partition_size

    class VocabParallelEmbedding(torch.nn.Embedding):
        """Megatron-compatible embedding for single-process inference."""

        def __init__(self, num_embeddings, embedding_dim, *, init_method=None, config=None, **kwargs):
            del config, kwargs
            super().__init__(num_embeddings, embedding_dim)
            self.tensor_model_parallel_size = 1
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings
            if init_method is not None:
                init_method(self.weight)

    class _LinearBase(torch.nn.Linear):
        def __init__(self, input_size, output_size, *, bias=True, init_method=None, config=None, **kwargs):
            del config, kwargs
            super().__init__(input_size, output_size, bias=bias)
            if init_method is not None:
                init_method(self.weight)

        def forward(self, input_: torch.Tensor, *args, **kwargs):
            del args, kwargs
            return super().forward(input_), None

    class ColumnParallelLinear(_LinearBase):
        """Megatron-compatible column linear for single-process inference."""

    class RowParallelLinear(_LinearBase):
        """Megatron-compatible row linear for single-process inference."""

    def reduce_from_tensor_model_parallel_region(tensor):
        return tensor

    def reduce_scatter_to_sequence_parallel_region(tensor):
        return tensor

    def model_parallel_cuda_manual_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "ColumnParallelLinear",
    "ModelParallelConfig",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabUtility",
    "model_parallel_cuda_manual_seed",
    "mpu",
    "parallel_state",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
]
