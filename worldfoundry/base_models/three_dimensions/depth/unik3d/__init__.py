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

"""Module for base_models -> three_dimensions -> depth -> unik3d -> __init__.py functionality."""

from typing import Literal

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType
from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .unik3d import UniK3D
from .utils.camera import Pinhole, Spherical


class Unik3DModel(DepthEstimationModel):
    """Unik d model implementation."""
    def __init__(self, type: Literal["s", "b", "l"] = "l", model_path: str | None = None) -> None:
        """Init.

        Args:
            type: The type.
            model_path: The model path.

        Returns:
            The return value.
        """
        super().__init__()
        self.model = UniK3D.from_pretrained(model_path or f"lpiccinelli/unik3d-vit{type}")
        self.model.resolution_level = 9
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        """Depth type.

        Returns:
            The return value.
        """
        return DepthType.MODEL_METRIC_DISTANCE

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return [CameraType.PINHOLE, CameraType.PANORAMA]

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Estimate.

        Args:
            src: The src.

        Returns:
            The return value.
        """
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = rgb.moveaxis(-1, 1) * 255.0
        H, W = rgb.shape[-2:]
        camera = None
        if src.camera_type == CameraType.PANORAMA:
            hfov2 = np.pi
            params = torch.tensor([0, 0, 0, 0, W, H, hfov2, H / W * hfov2], device=rgb.device).float()
            camera = Spherical(params=params)
        elif src.camera_type == CameraType.PINHOLE and src.intrinsics is not None:
            focal = src.intrinsics.reshape(-1).to(device=rgb.device, dtype=torch.float32)
            if focal.numel() == 1:
                focal = focal.repeat(rgb.shape[0])
            K = torch.eye(3, device=rgb.device).unsqueeze(0).repeat(rgb.shape[0], 1, 1)
            K[:, 0, 0] = focal
            K[:, 1, 1] = focal
            K[:, 0, 2] = W / 2
            K[:, 1, 2] = H / 2
            camera = Pinhole(K=K)
        elif src.camera_type != CameraType.PINHOLE:
            raise ValueError(f"Unsupported UniK3D camera type: {src.camera_type.value}")
        outputs = self.model.infer(rgb, camera=camera, normalize=True)

        pred_distance = outputs["distance"].squeeze(1)
        confidence = outputs["confidence"].squeeze(1)
        points = outputs["points"].moveaxis(1, -1)

        if not batch_dim:
            pred_distance, confidence, points = pred_distance[0], confidence[0], points[0]

        return DepthEstimationResult(
            metric_depth=pred_distance,
            confidence=confidence,
            points=points,
        )
