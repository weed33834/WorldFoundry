# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyCall as L
from worldfoundry.synthesis.visual_generation.gamma_world.models.causal_multiview import (
    CausalJointCosmosMVModel,
    CausalJointCosmosMVModelConfig,
)

FSDP_CONFIG_MV = dict(
    model=L(CausalJointCosmosMVModel)(
        config=CausalJointCosmosMVModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_mv", node=FSDP_CONFIG_MV)
