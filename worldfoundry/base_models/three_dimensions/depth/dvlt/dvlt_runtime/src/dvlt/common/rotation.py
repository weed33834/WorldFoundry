# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

"""Rotation helpers used by DVLT."""

import torch
from torch import Tensor

from worldfoundry.core.geometry import (
    quaternion_xyzw_to_rotation_matrix as quat_to_mat,
    rotation_matrix_to_quaternion_xyzw as mat_to_quat,
    standardize_quaternion_xyzw as standardize_quaternion,
)


def so3_relative_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """Compute the angle between two batches of rotation matrices."""

    cosine = (torch.einsum("...ii", R1.mT @ R2) - 1) / 2
    cosine = torch.clip(cosine, -1.0, 1.0)
    return torch.abs(torch.arccos(cosine))


def quaternion_slerp(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Spherically interpolate two XYZW quaternions."""

    q1 = standardize_quaternion(q1)
    q2 = standardize_quaternion(q2)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = torch.clamp((q1 * q2).sum(dim=-1, keepdim=True), -1.0, 1.0)
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    small_angle = sin_theta_0.abs() < 1e-8
    s1 = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = torch.sin(t * theta_0) / sin_theta_0
    out_slerp = s1 * q1 + s2 * q2
    out_lerp = (1.0 - t) * q1 + t * q2
    return standardize_quaternion(torch.where(small_angle, out_lerp, out_slerp))
