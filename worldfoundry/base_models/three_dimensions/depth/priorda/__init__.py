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

"""Module for base_models -> three_dimensions -> depth -> priorda -> __init__.py functionality."""

import torch
from einops import rearrange

from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .priorda import PriorDepthAnything


class PriorDAModel(DepthEstimationModel):
    """
    https://github.com/SpatialVision/Prior-Depth-Anything
    """

    def __init__(self, weights_dir: str | None = None, device: str = "cuda") -> None:
        """Init.

        Args:
            weights_dir: The weights dir.
            device: The device.

        Returns:
            The return value.
        """
        super().__init__()
        self.model = PriorDepthAnything(
            device=device,
            fmde_dir=weights_dir,
            cmde_dir=weights_dir,
            ckpt_dir=weights_dir,
        )

    @property
    def depth_type(self) -> DepthType:
        """Depth type.

        Returns:
            The return value.
        """
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput, *, pattern: str | None = None) -> DepthEstimationResult:
        """Estimate.

        Args:
            src: The src.

        Returns:
            The return value.
        """
        rgb: torch.Tensor = unpack_optional(src.rgb)
        prompt_metric_depth: torch.Tensor = unpack_optional(src.prompt_metric_depth)

        assert rgb.dim() == 3 and prompt_metric_depth.dim() == 2, "Single batch only"
        final_depth = self.model.infer_one_sample(
            image=rgb * 255.0,
            prior=prompt_metric_depth,
            geometric=None,
            pattern=pattern,
        )

        return DepthEstimationResult(metric_depth=final_depth)
