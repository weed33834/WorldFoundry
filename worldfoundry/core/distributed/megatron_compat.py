"""Compatibility wrapper for optional Megatron model-parallel state."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from megatron.core import ModelParallelConfig, mpu, parallel_state
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


__all__ = ["ModelParallelConfig", "mpu", "parallel_state"]
