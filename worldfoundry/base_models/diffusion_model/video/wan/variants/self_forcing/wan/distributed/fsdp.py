# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> variants -> self_forcing -> wan -> distributed -> fsdp.py functionality."""

from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    """Shard model.

    Args:
        model: The model.
        device_id: The device id.
        param_dtype: The param dtype.
        reduce_dtype: The reduce dtype.
        buffer_dtype: The buffer dtype.
        process_group: The process group.
        sharding_strategy: The sharding strategy.
        sync_module_states: The sync module states.
    """
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        use_orig_params=True,
        sync_module_states=sync_module_states)
    return model
