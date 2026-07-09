# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:  # Torch < 2.9 lacks the FSDP2 helper used upstream.
    MixedPrecisionPolicy = None
    fully_shard = None

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    if fully_shard is None or MixedPrecisionPolicy is None:
        model.to(param_dtype)
        if torch.cuda.is_available():
            model.to(torch.device("cuda", torch.cuda.current_device()))
        return model

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
