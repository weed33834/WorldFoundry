import importlib.util

import torch
import torch.distributed as dist

try:
    # The pai_fuser is an internally developed acceleration package, which can be used on PAI.
    if importlib.util.find_spec("paifuser") is not None:
        from paifuser.xfuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size,
            get_sp_group, get_world_group, init_distributed_environment,
            initialize_model_parallel, model_parallel_is_initialized)
        from paifuser.xfuser.core.long_ctx_attention import \
            xFuserLongContextAttention
        print("Import PAI DiT Turbo")
    else:
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group, get_world_group,
                                             init_distributed_environment,
                                             initialize_model_parallel,
                                             model_parallel_is_initialized)
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        print("xFuser available")
except Exception:
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None
    model_parallel_is_initialized = None


def set_multi_gpus_devices(ulysses_degree, ring_degree, classifier_free_guidance_degree=1):
    if ulysses_degree > 1 or ring_degree > 1 or classifier_free_guidance_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xfuser is not installed.")
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        print(
            "parallel inference enabled: ulysses_degree=%d ring_degree=%d "
            "classifier_free_guidance_degree=%d rank=%d world_size=%d"
            % (
                ulysses_degree,
                ring_degree,
                classifier_free_guidance_degree,
                dist.get_rank(),
                dist.get_world_size(),
            )
        )
        expected_world_size = (
            ring_degree * ulysses_degree * classifier_free_guidance_degree
        )
        if dist.get_world_size() != expected_world_size:
            raise ValueError(
                f"World size {dist.get_world_size()} does not match the requested "
                f"parallel degree {expected_world_size}."
            )
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=ring_degree * ulysses_degree,
                classifier_free_guidance_degree=classifier_free_guidance_degree,
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree)
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        # ``torch.cuda.set_device(torch.device("cuda"))`` is rejected by
        # current PyTorch because the device has no explicit index.  Return
        # the process-local CUDA device just like the distributed branch;
        # with CUDA_VISIBLE_DEVICES this remains logical ``cuda:0``.
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def sequence_parallel_chunk(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x

    sp_rank = get_sequence_parallel_rank()
    if x.size(dim) % sp_world_size != 0:
        raise ValueError(
            f"Dimension {dim} of x ({x.size(dim)}) is not divisible by "
            f"sequence-parallel world size {sp_world_size}."
        )

    chunks = torch.chunk(x, sp_world_size, dim=dim)
    x = chunks[sp_rank]

    return x


def sequence_parallel_all_gather(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x  # No gathering needed

    sp_group = get_sp_group()
    return sp_group.all_gather(x, dim=dim)


__all__ = [
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sp_group",
    "get_world_group",
    "init_distributed_environment",
    "initialize_model_parallel",
    "model_parallel_is_initialized",
    "sequence_parallel_all_gather",
    "sequence_parallel_chunk",
    "set_multi_gpus_devices",
    "xFuserLongContextAttention",
]
