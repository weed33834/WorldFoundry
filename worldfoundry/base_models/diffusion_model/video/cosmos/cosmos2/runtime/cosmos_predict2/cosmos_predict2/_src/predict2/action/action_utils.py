"""Action-state conversion used by action-conditioned inference entry points."""

import numpy as np

from worldfoundry.core.geometry import (
    euler_angles_to_rotation_matrix_zyx,
    rotation_matrix_to_euler_angles_zyx,
    rotation_matrix_to_quaternion_wxyz,
)


def get_action_sequence_from_states(
    data: dict,
    *,
    fps_downsample_ratio: int = 1,
    use_quat: bool = False,
    state_key: str = "state",
    gripper_scale: float = 1.0,
    gripper_key: str = "continuous_gripper_state",
    action_scale: float = 20.0,
) -> np.ndarray:
    """Convert absolute robot states to scaled relative actions."""
    arm_states = np.asarray(data[state_key])[::fps_downsample_ratio]
    gripper_states = np.asarray(data[gripper_key])[::fps_downsample_ratio]
    if len(arm_states) != len(gripper_states):
        raise ValueError("robot and gripper state sequences must have equal length")

    rotation_dim = 4 if use_quat else 3
    actions = np.zeros((max(len(arm_states) - 1, 0), 3 + rotation_dim + 1), dtype=np.float64)
    for index in range(1, len(arm_states)):
        previous_rotation = euler_angles_to_rotation_matrix_zyx(arm_states[index - 1, 3:6])
        current_rotation = euler_angles_to_rotation_matrix_zyx(arm_states[index, 3:6])
        relative_rotation = previous_rotation.T @ current_rotation
        actions[index - 1, :3] = previous_rotation.T @ (arm_states[index, :3] - arm_states[index - 1, :3])
        if use_quat:
            actions[index - 1, 3:7] = rotation_matrix_to_quaternion_wxyz(relative_rotation)
        else:
            actions[index - 1, 3:6] = rotation_matrix_to_euler_angles_zyx(relative_rotation)
        actions[index - 1, -1] = gripper_states[index]

    rotation_scales = [action_scale] * rotation_dim
    actions *= np.asarray([action_scale] * 3 + rotation_scales + [gripper_scale])
    return actions


__all__ = ["get_action_sequence_from_states"]
