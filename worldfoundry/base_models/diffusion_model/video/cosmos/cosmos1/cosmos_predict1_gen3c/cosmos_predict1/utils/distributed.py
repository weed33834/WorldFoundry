# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for GEN3C distributed helpers.

The vendored GEN3C inference runtime only needs a subset of the official
``cosmos_predict1.utils.distributed`` module. Re-export the shared WorldFoundry
helpers to avoid pulling in optional training-time dependencies at import time.
"""

from worldfoundry.core.distributed.torch_process_group import (
    barrier,
    device_with_rank,
    get_rank,
    get_world_size,
    init,
    is_local_rank0,
    is_rank0,
    rank0_first,
    rank0_only,
)

__all__ = [
    "barrier",
    "device_with_rank",
    "get_rank",
    "get_world_size",
    "init",
    "is_local_rank0",
    "is_rank0",
    "rank0_first",
    "rank0_only",
]
