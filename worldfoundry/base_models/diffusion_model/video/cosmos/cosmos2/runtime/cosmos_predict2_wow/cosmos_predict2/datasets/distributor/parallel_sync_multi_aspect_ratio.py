# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2_wow -> cosmos_predict2 -> datasets -> distributor -> parallel_sync_multi_aspect_ratio.py functionality."""

from imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
from imaginaire.utils import log

from worldfoundry.core.distributed.parallel_shard_iterator import ParallelSyncMultiAspectRatioMixin


class ShardlistMultiAspectRatioParallelSync(ParallelSyncMultiAspectRatioMixin, ShardlistMultiAspectRatio):
    """Shardlist multi aspect ratio parallel sync implementation."""
    log = log

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.enable_parallel()
