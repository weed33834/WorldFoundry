# Shared Sequence Parallel (SP) infrastructure.
# Re-exports the core API from the in-tree Wan base-model package.

from .communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from .parallel_state import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    get_local_torch_device,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from .utils import (
    StatelessProcessGroup,
    divide,
    ensure_divisibility,
    split_tensor_along_last_dim,
)

__all__ = [
    "StatelessProcessGroup",
    "cleanup_dist_env_and_memory",
    "divide",
    "ensure_divisibility",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "get_local_torch_device",
    "get_sp_group",
    "get_sp_parallel_rank",
    "get_sp_world_size",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "get_world_group",
    "get_world_rank",
    "get_world_size",
    "init_distributed_environment",
    "initialize_model_parallel",
    "maybe_init_distributed_environment_and_model_parallel",
    "model_parallel_is_initialized",
    "sequence_model_parallel_all_gather",
    "sequence_model_parallel_all_to_all_4D",
    "split_tensor_along_last_dim",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
]
