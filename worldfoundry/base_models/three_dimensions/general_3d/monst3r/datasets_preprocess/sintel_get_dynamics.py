"""Minimal MonST3R optical-flow geometry helper.

The upstream MonST3R demo imports ``compute_optical_flow`` from its dataset
preprocessing package while exporting optional dynamic masks.  WorldFoundry only
vendors the infer-time geometry helper, not the dataset conversion scripts.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _depth_to_3d(depth_map, intrinsic_matrix):
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    x = (i - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (j - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map
    return np.stack([x, y, z], axis=-1)


def _project_3d_to_2d(points_3d, intrinsic_matrix):
    projected_2d_hom = intrinsic_matrix @ points_3d.T
    return (projected_2d_hom[:2, :] / projected_2d_hom[2, :]).T


def compute_optical_flow(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2):
    """Project frame-1 depth into frame 2 and return the induced optical flow."""
    del depth2
    depth1 = _to_numpy(depth1)
    pose1 = _to_numpy(pose1)
    pose2 = _to_numpy(pose2)
    intrinsic_matrix1 = _to_numpy(intrinsic_matrix1)
    intrinsic_matrix2 = _to_numpy(intrinsic_matrix2)

    points_3d_frame1 = _depth_to_3d(depth1, intrinsic_matrix1).reshape(-1, 3)
    points_3d_frame1_hom = np.concatenate(
        [points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))],
        axis=1,
    ).T
    transformation_matrix = pose2 @ np.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = points_3d_frame2_hom[:3, :].T

    points_2d_frame1 = _project_3d_to_2d(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2 = _project_3d_to_2d(points_3d_frame2, intrinsic_matrix2)
    return points_2d_frame2 - points_2d_frame1
