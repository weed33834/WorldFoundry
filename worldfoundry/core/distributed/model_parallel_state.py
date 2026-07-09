"""Small compatibility helpers around Megatron model-parallel state."""

from __future__ import annotations

from worldfoundry.core.distributed.megatron_compat import parallel_state


def is_tp_cp_pp_rank0() -> bool:
    """Return true when tensor, context, and pipeline parallel ranks are all zero."""

    return (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
        and parallel_state.get_context_parallel_rank() == 0
    )


__all__ = ["is_tp_cp_pp_rank0"]
