# Inference-only PLY export adapted from the official Splatt3R demo utilities.
"""Module for base_models -> three_dimensions -> general_3d -> splatt3r -> splatt3r_runtime -> utils -> export.py functionality."""

from __future__ import annotations

import einops
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def _inverse_sigmoid(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert activated opacity probabilities to the logits stored by 3DGS PLY."""

    values = values.clamp(min=eps, max=1.0 - eps)
    return torch.log(values) - torch.log1p(-values)


def _covariance_to_quaternion_and_scale(covariance: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Factor symmetric covariances into standard 3DGS ``wxyz`` rotations and scales."""

    covariance = 0.5 * (covariance + covariance.transpose(-2, -1))
    eigenvalues, rotation = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)

    # Eigenvectors can form an improper rotation (det=-1).  Flip one axis while
    # keeping R diag(s^2) R^T unchanged so scipy receives a proper rotation.
    improper = torch.linalg.det(rotation) < 0
    if improper.any():
        rotation = rotation.clone()
        rotation[improper, :, -1] *= -1

    scale = torch.sqrt(eigenvalues).detach().cpu().numpy()
    quaternion_xyzw = Rotation.from_matrix(rotation.detach().cpu().numpy()).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    return quaternion_wxyz, scale


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

    means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    scales = torch.stack([pred1["scales"], pred2["scales"]], dim=1)
    rotations = torch.stack([pred1["rotations"], pred2["rotations"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]
    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

    means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    scales = einops.rearrange(scales[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    rotations = einops.rearrange(rotations[0], "view h w xyzw -> (view h w) xyzw")
    rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rotations = rotations[:, [3, 0, 1, 2]].detach().cpu().numpy()
    harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz")
    opacities = _inverse_sigmoid(opacities).detach().cpu().numpy()

    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(save_path)


__all__ = ["save_as_ply"]
