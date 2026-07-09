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

"""Module for base_models -> three_dimensions -> general_3d -> vipe -> slam -> maths -> retractor.py functionality."""

from typing import Any

import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.lietorch import SE3
from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType


class BaseRetractor:
    """Base retractor implementation."""
    def oplus(self, x: Any, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        x[inds] += dx


class PoseRetractor(BaseRetractor):
    """Pose retractor implementation."""
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class RigRotationOnlyRetractor(BaseRetractor):
    """Rig rotation only retractor implementation."""
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        dx = dx.clone()
        dx[:, :3] = 0  # zero out translation part
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class DenseDispRetractor(BaseRetractor):
    """Dense disp retractor implementation."""
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        dx = torch.where(dx > 10, torch.zeros_like(dx), dx)
        super().oplus(x, inds, dx)


class TracksDispRetractor(BaseRetractor):
    """Tracks disp retractor implementation."""
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        super().oplus(x, inds, dx)
        x.clamp_(min=1e-3, max=10)


class IntrinsicsRetractor(BaseRetractor):
    """Intrinsics retractor implementation."""
    def __init__(self, camera_type: CameraType):
        """Init.

        Args:
            camera_type: The camera type.
        """
        self.camera_type = camera_type

    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        """Oplus.

        Args:
            x: The x.
            inds: The inds.
            dx: The dx.

        Returns:
            The return value.
        """
        if len(dx) == 1:
            # Broadcast dx to all intrinsics
            inds = torch.where(x[:, 0] > 0)[0]
            dx = dx.repeat(len(inds), 1)
        x[inds, :2] += dx[..., :1]
        # Use smaller learning rate for the distortion parameters
        x[inds, 4:] += dx[..., 1:] * 0.01
