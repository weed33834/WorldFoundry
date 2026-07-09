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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2_wow -> cosmos_predict2 -> tokenizers -> interface.py functionality."""

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch


class VideoTokenizerInterface(ABC):
    """Video tokenizer interface implementation."""
    def __init__(self):
        """Init."""
        pass

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode.

        Args:
            state: The state.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode.

        Args:
            latent: The latent.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """Get latent num frames.

        Args:
            num_pixel_frames: The num pixel frames.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """Get pixel num frames.

        Args:
            num_latent_frames: The num latent frames.

        Returns:
            The return value.
        """
        pass

    @property
    @abstractmethod
    def spatial_compression_factor(self):
        """Spatial compression factor."""
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self):
        """Temporal compression factor."""
        pass

    @property
    @abstractmethod
    def spatial_resolution(self):
        """Spatial resolution."""
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        """Pixel chunk duration."""
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        """Latent chunk duration."""
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        """Latent ch.

        Returns:
            The return value.
        """
        pass

    @property
    def is_chunk_overlap(self):
        """Is chunk overlap."""
        return False

    @property
    def is_causal(self):
        """Is causal."""
        return True
