"""Pure tensor transforms used when merging independently trained 4DGS scenes."""

import torch


def rotation_matrix_from_angles(rotation_angles: torch.Tensor) -> torch.Tensor:
    """Build ``Rz(theta) @ Rx(phi)`` without copying device scalars to CPU."""

    theta, phi = rotation_angles.unbind()
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    rotation_z = torch.stack(
        (
            torch.stack((cos_theta, -sin_theta, zero)),
            torch.stack((sin_theta, cos_theta, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    rotation_x = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, cos_phi, -sin_phi)),
            torch.stack((zero, sin_phi, cos_phi)),
        )
    )
    return rotation_z @ rotation_x


def rotate_point_cloud(
    point_cloud: torch.Tensor,
    displacement: torch.Tensor,
    rotation_angles: torch.Tensor,
    scales_bias,
) -> torch.Tensor:
    """Apply the merge script's scale, rotation, and translation to row points."""

    rotation_matrix = rotation_matrix_from_angles(rotation_angles)
    return (point_cloud * scales_bias) @ rotation_matrix.transpose(-1, -2) + displacement


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for broadcast-compatible quaternions in ``wxyz`` order."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_from_angles_wxyz(rotation_angles: torch.Tensor) -> torch.Tensor:
    """Return the quaternion for ``Rz(theta) @ Rx(phi)`` in ``wxyz`` order."""

    theta, phi = rotation_angles.unbind()
    zero = torch.zeros_like(theta)
    qz = torch.stack(
        (torch.cos(theta / 2), zero, zero, torch.sin(theta / 2))
    )
    qx = torch.stack(
        (torch.cos(phi / 2), torch.sin(phi / 2), zero, zero)
    )
    return quaternion_multiply_wxyz(qz, qx)


def apply_rotation_bias_wxyz(
    rotations: torch.Tensor, rotation_angles: torch.Tensor
) -> torch.Tensor:
    """Left-multiply Gaussian orientations by the scene rotation and normalize."""

    rotation_bias = quaternion_from_angles_wxyz(rotation_angles)
    transformed = quaternion_multiply_wxyz(rotation_bias, rotations)
    eps = torch.finfo(transformed.dtype).eps
    return transformed / torch.linalg.vector_norm(
        transformed, dim=-1, keepdim=True
    ).clamp_min(eps)
