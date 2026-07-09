# Inference-only PLY export adapted from the official Splatt3R demo utilities.
"""Module for base_models -> three_dimensions -> general_3d -> splatt3r -> splatt3r_runtime -> utils -> export.py functionality."""

from __future__ import annotations

import einops
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def save_as_ply(pred1, pred2, save_path: str) -> None:
    """Save as ply.

    Args:
        pred1: The pred1.
        pred2: The pred2.
        save_path: The save path.

    Returns:
        The return value.
    """
    def construct_list_of_attributes(num_rest: int) -> list[str]:
        """Construct list of attributes.

        Args:
            num_rest: The num rest.

        Returns:
            The return value.
        """
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        """Covariance to quaternion and scale.

        Args:
            covariance: The covariance.
        """
        u, s, v = torch.linalg.svd(covariance)
        scale = torch.sqrt(s).detach().cpu().numpy()
        rotation_matrix = torch.bmm(u, v.transpose(-2, -1)).detach().cpu().numpy()
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        return quaternion, scale

    means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]
    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

    means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
    harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()

    rotations, scales = covariance_to_quaternion_and_scale(covariances)
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(save_path)


__all__ = ["save_as_ply"]
