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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> predict2 -> configs -> video2world -> config.py functionality."""

from typing import Any, List

import attrs
from cosmos_predict2._src.predict2.configs.video2world.inference_defaults import register_inference_defaults
from cosmos_predict2._src.predict2.configs.video2world.released import register_released_experiments

from worldfoundry.core.configuration import Config as InferenceConfig


@attrs.define(slots=False)
class Config(InferenceConfig):
    """Config implementation."""

    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "fsdp_rectified_flow"},
            {"net": "cosmos_v1_2B"},
            {"conditioner": "video_prediction_conditioner"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    """Make config.

    Returns:
        The return value.
    """
    c = Config(model=None)

    # Specifying values through instances of attrs
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    register_inference_defaults()
    register_released_experiments()

    return c
