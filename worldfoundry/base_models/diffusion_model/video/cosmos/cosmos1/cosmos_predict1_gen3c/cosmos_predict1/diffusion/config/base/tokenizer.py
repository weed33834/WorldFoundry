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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> diffusion -> config -> base -> tokenizer.py functionality."""

import omegaconf

from cosmos_predict1.diffusion.module.pretrained_vae import JITVAE, JointImageVideoSharedJITTokenizer, VideoJITTokenizer
from cosmos_predict1.utils.lazy_config import LazyCall as L

TOKENIZER_OPTIONS = {}


def tokenizer_register(key):
    """Tokenizer register.

    Args:
        key: The key.
    """
    def decorator(func):
        """Decorator.

        Args:
            func: The func.
        """
        TOKENIZER_OPTIONS[key] = func
        return func

    return decorator


@tokenizer_register("cosmos_diffusion_tokenizer_comp8x8x8")
def get_cosmos_diffusion_tokenizer_comp8x8x8(resolution: str, chunk_duration: int) -> omegaconf.dictconfig.DictConfig:
    """Get cosmos diffusion tokenizer comp8x8x8.

    Args:
        resolution: The resolution.
        chunk_duration: The chunk duration.

    Returns:
        The return value.
    """
    assert resolution in ["720"]

    pixel_chunk_duration = chunk_duration
    temporal_compression_factor = 8
    spatial_compression_factor = 8

    return L(JointImageVideoSharedJITTokenizer)(
        video_vae=L(VideoJITTokenizer)(
            name="cosmos_predict1_tokenizer",
            latent_ch=16,
            is_bf16=True,
            pixel_chunk_duration=pixel_chunk_duration,
            temporal_compression_factor=temporal_compression_factor,
            spatial_compression_factor=spatial_compression_factor,
            spatial_resolution=resolution,
        ),
        image_vae=L(JITVAE)(
            name="cosmos_predict1_tokenizer",
            latent_ch=16,
            is_image=False,
            is_bf16=True,
        ),
        name="cosmos_predict1_tokenizer",
        latent_ch=16,
    )
