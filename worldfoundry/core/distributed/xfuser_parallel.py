"""xFuser-backed sequence and tensor parallel helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist
import xfuser


def initialize_parallel_group(ring_degree, ulysses_degree, tensor_parallel_degree):
    dist.init_process_group("nccl")
    xfuser.core.distributed.init_distributed_environment(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    xfuser.core.distributed.initialize_model_parallel(
        sequence_parallel_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
        tensor_parallel_degree=tensor_parallel_degree,
    )
    torch.cuda.set_device(dist.get_rank())


def initialize_parall_group(ring_degree, ulysses_degree, tensor_parallel_degree):
    """Backward-compatible alias for the historical misspelled name."""

    return initialize_parallel_group(ring_degree, ulysses_degree, tensor_parallel_degree)


def get_parallel_group():
    return xfuser.core.distributed.get_world_group()


def get_sequence_parallel_world_size():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()


def get_sequence_parallel_rank():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()


def get_sp_group():
    return xfuser.core.distributed.parallel_state.get_sp_group()


def parallel_forward(fn_):
    def wrapped(_, hidden_states, *args, **kwargs):
        if kwargs["parallel"]:
            hidden_states = torch.chunk(
                hidden_states,
                get_sequence_parallel_world_size(),
                dim=-2,
            )[get_sequence_parallel_rank()]
            kwargs["attn_mask"] = torch.chunk(
                kwargs["attn_mask"],
                get_sequence_parallel_world_size(),
                dim=-2,
            )[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)

        if kwargs["parallel"]:
            output = get_sp_group().all_gather(output.contiguous(), dim=-2)
        return output

    return wrapped


__all__ = [
    "get_parallel_group",
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sp_group",
    "initialize_parall_group",
    "initialize_parallel_group",
    "parallel_forward",
]
