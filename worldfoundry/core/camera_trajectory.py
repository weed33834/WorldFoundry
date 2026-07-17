"""Reusable camera-trajectory parsing and action-token conversion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from worldfoundry.core.geometry import rotation_matrix_to_euler_angles_zyx

_TRANSLATION_LABELS = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

_TRAJECTORY_ACTION_LABELS = {
    "w": (1, 0),
    "s": (2, 0),
    "a": (3, 0),
    "d": (4, 0),
    "j": (0, 2),
    "l": (0, 1),
    "i": (0, 3),
    "k": (0, 4),
}


def _rotation_x(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)


def parse_camera_trajectory(
    trajectory: str,
    *,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
) -> list[dict[str, float]]:
    """Parse ``w*4,j*2`` camera controls into per-frame motions."""

    rotation_step = np.radians(float(rotation_step_degrees))
    motions = {
        "w": {"forward": float(translation_step)},
        "s": {"forward": -float(translation_step)},
        "d": {"right": float(translation_step)},
        "a": {"right": -float(translation_step)},
        "u": {"up": float(translation_step)},
        "dn": {"up": -float(translation_step)},
        "j": {"yaw": -rotation_step},
        "l": {"yaw": rotation_step},
        "i": {"pitch": rotation_step},
        "k": {"pitch": -rotation_step},
        "left": {"yaw": -rotation_step},
        "right": {"yaw": rotation_step},
        "up": {"pitch": rotation_step},
        "down": {"pitch": -rotation_step},
        "z": {},
    }
    parsed: list[dict[str, float]] = []
    for raw_segment in str(trajectory).strip().split(","):
        segment = raw_segment.strip().lower()
        if not segment:
            continue
        match = re.fullmatch(r"([a-z]+)(?:\*(\d+))?", segment)
        if match is None:
            raise ValueError(f"Invalid camera trajectory segment {raw_segment!r}; expected e.g. 'w*19'")
        key, raw_count = match.groups()
        if key not in motions:
            raise ValueError(f"Unknown camera direction {key!r}; choices: {sorted(motions)}")
        count = int(raw_count or 1)
        if count < 0:
            raise ValueError(f"Camera trajectory count must be non-negative, got {count}")
        parsed.extend(dict(motions[key]) for _ in range(count))
    return parsed


def camera_trajectory_view_matrices(
    trajectory: str | Sequence[Mapping[str, float]],
    *,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
) -> np.ndarray:
    """Return OpenCV world-to-camera matrices, including the identity frame."""

    motions = (
        parse_camera_trajectory(
            trajectory,
            translation_step=translation_step,
            rotation_step_degrees=rotation_step_degrees,
        )
        if isinstance(trajectory, str)
        else [dict(item) for item in trajectory]
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    poses = [camera_to_world.copy()]
    for motion in motions:
        if "yaw" in motion:
            camera_to_world[:3, :3] = camera_to_world[:3, :3] @ _rotation_y(float(motion["yaw"]))
        if "pitch" in motion:
            camera_to_world[:3, :3] = camera_to_world[:3, :3] @ _rotation_x(float(motion["pitch"]))
        local_translation = np.asarray(
            [float(motion.get("right", 0.0)), -float(motion.get("up", 0.0)), float(motion.get("forward", 0.0))]
        )
        camera_to_world[:3, 3] += camera_to_world[:3, :3] @ local_translation
        poses.append(camera_to_world.copy())
    return np.stack([np.linalg.inv(pose) for pose in poses]).astype(np.float32)


def camera_trajectory_tensors(
    trajectory: str | Sequence[Mapping[str, float]],
    *,
    fx: float = 0.5050505,
    fy: float = 0.89786756,
    cx: float = 0.5,
    cy: float = 0.5,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
    device: Any = "cpu",
    dtype: Any = None,
):
    """Return batched view matrices and normalized camera intrinsics."""

    import torch

    tensor_dtype = dtype or torch.float32
    view_matrices = camera_trajectory_view_matrices(
        trajectory,
        translation_step=translation_step,
        rotation_step_degrees=rotation_step_degrees,
    )
    intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.repeat(intrinsic[None], len(view_matrices), axis=0)
    return (
        torch.as_tensor(view_matrices, dtype=tensor_dtype, device=device).unsqueeze(0),
        torch.as_tensor(intrinsics, dtype=tensor_dtype, device=device).unsqueeze(0),
    )


def camera_poses_to_adaln_actions(
    camera_to_world: Any,
    *,
    action_scale: str | Sequence[float],
    temporal_stride: int = 8,
):
    """Convert pixel-rate camera poses to latent-rate 6D AdaLN actions.

    The returned tensor has shape ``[B, T_latent, 6]`` and stores consecutive
    camera-local ``[tx, ty, tz, rx, ry, rz]`` deltas.  Frame zero is all zeros.
    This is the common representation used by AlayaWorld's action conditioner;
    keeping it in :mod:`worldfoundry.core` also avoids a SciPy runtime
    dependency for camera-conditioned world models.

    Args:
        camera_to_world: ``[F, 4, 4]`` or ``[B, F, 4, 4]`` camera-to-world
            matrices. Torch tensors are accepted and converted back to Torch.
        action_scale: Six positive normalization constants, either a
            comma-separated string or a sequence.
        temporal_stride: Pixel frames represented by one video latent frame.
    """

    import torch

    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    is_tensor = isinstance(camera_to_world, torch.Tensor)
    source_device = camera_to_world.device if is_tensor else None
    poses = camera_to_world.detach().cpu().numpy() if is_tensor else np.asarray(camera_to_world)
    if poses.ndim == 3:
        poses = poses[None]
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must be [F,4,4] or [B,F,4,4], got {poses.shape}")
    if poses.shape[1] < 1:
        raise ValueError("camera_to_world must contain at least one frame")

    raw_scale = action_scale.split(",") if isinstance(action_scale, str) else action_scale
    try:
        scale = np.asarray([float(value) for value in raw_scale], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("action_scale must contain six positive floats") from exc
    if scale.shape != (6,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"action_scale must contain six positive floats, got {action_scale!r}")

    result: list[np.ndarray] = []
    for batch_poses in poses.astype(np.float64, copy=False):
        latent_count = (len(batch_poses) - 1) // temporal_stride + 1
        sampled = batch_poses[np.minimum(np.arange(latent_count) * temporal_stride, len(batch_poses) - 1)]
        actions = np.zeros((latent_count, 6), dtype=np.float32)
        for index in range(1, latent_count):
            relative = np.linalg.inv(sampled[index - 1]) @ sampled[index]
            actions[index, :3] = relative[:3, 3]
            actions[index, 3:] = rotation_matrix_to_euler_angles_zyx(relative[:3, :3])
        result.append(actions / scale[None])

    stacked = np.stack(result)
    if is_tensor:
        return torch.as_tensor(stacked, dtype=torch.float32, device=source_device)
    return stacked


def select_adaln_actions(actions: Any, latent_indices: Any, *, device: Any = None, dtype: Any = None):
    """Select and clamp latent-rate action rows for a rollout segment."""

    import torch

    values = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError(f"actions must be [B,T,6], got {tuple(values.shape)}")
    target_device = device if device is not None else values.device
    indices = torch.as_tensor(latent_indices, device=target_device, dtype=torch.long).flatten()
    indices = indices.clamp(min=0, max=values.shape[1] - 1)
    values = values.to(device=target_device, dtype=dtype or values.dtype)
    return values.index_select(1, indices)


def one_hot_camera_actions_to_labels(one_hot: Any) -> np.ndarray:
    """Map forward/backward/left/right one-hot rows to labels 0–8."""

    rows = np.asarray(one_hot)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"Expected camera action rows shaped (N, 4), got {rows.shape}")
    return np.asarray([_TRANSLATION_LABELS.get(tuple(int(value) for value in row), 0) for row in rows], dtype=np.int64)


def discretize_camera_poses_to_actions(view_matrices: Any) -> np.ndarray:
    """Derive the 81-class WorldPlay action labels from world-to-camera poses."""

    from scipy.spatial.transform import Rotation

    matrices = np.asarray(view_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera matrices shaped (T, 4, 4), got {matrices.shape}")
    camera_to_world = np.linalg.inv(matrices)
    count = len(matrices)
    translation = np.zeros((count, 4), dtype=np.int32)
    rotation = np.zeros((count, 4), dtype=np.int32)
    for index in range(1, count):
        relative = np.linalg.inv(camera_to_world[index - 1]) @ camera_to_world[index]
        direction = relative[:3, 3]
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            angles = np.degrees(np.arccos(np.clip(direction / norm, -1.0, 1.0)))
            if angles[2] < 60:
                translation[index, 0] = 1
            elif angles[2] > 120:
                translation[index, 1] = 1
            if angles[0] < 60:
                translation[index, 2] = 1
            elif angles[0] > 120:
                translation[index, 3] = 1
        rotation_angles = Rotation.from_matrix(relative[:3, :3]).as_euler("xyz", degrees=True)
        if rotation_angles[1] > 0.05:
            rotation[index, 0] = 1
        elif rotation_angles[1] < -0.05:
            rotation[index, 1] = 1
        if rotation_angles[0] > 0.05:
            rotation[index, 2] = 1
        elif rotation_angles[0] < -0.05:
            rotation[index, 3] = 1
    return one_hot_camera_actions_to_labels(translation) * 9 + one_hot_camera_actions_to_labels(rotation)


def camera_trajectory_action_labels(trajectory: str, num_frames: int):
    """Convert a compact trajectory string to a padded int64 Torch tensor."""

    import torch

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    labels: list[int] = []
    for raw_segment in str(trajectory).strip().split(","):
        segment = raw_segment.strip().lower()
        if not segment:
            continue
        match = re.fullmatch(r"([a-z]+)(?:\*(\d+))?", segment)
        if match is None:
            raise ValueError(f"Invalid camera trajectory segment {raw_segment!r}")
        key, raw_count = match.groups()
        translation_label, rotation_label = _TRAJECTORY_ACTION_LABELS.get(key, (0, 0))
        labels.extend([translation_label * 9 + rotation_label] * int(raw_count or 1))
    result = np.zeros(num_frames, dtype=np.int64)
    fill_length = min(len(labels), num_frames - 1)
    result[1 : 1 + fill_length] = labels[:fill_length]
    return torch.from_numpy(result)


__all__ = [
    "camera_poses_to_adaln_actions",
    "camera_trajectory_action_labels",
    "camera_trajectory_tensors",
    "camera_trajectory_view_matrices",
    "discretize_camera_poses_to_actions",
    "one_hot_camera_actions_to_labels",
    "parse_camera_trajectory",
    "select_adaln_actions",
]
