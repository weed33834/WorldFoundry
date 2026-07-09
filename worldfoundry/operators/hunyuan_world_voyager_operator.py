"""Module for the HunyuanWorldVoyager operator implementation."""

import numpy as np
from PIL import Image
import torch

from .base_operator import BaseOperator


_INTERACTION_ALIASES = {
    "turn_left": "camera_l",
    "turn-left": "camera_l",
    "camera-left": "camera_l",
    "camera-left-turn": "camera_l",
    "turn_right": "camera_r",
    "turn-right": "camera_r",
    "camera-right": "camera_r",
    "camera-right-turn": "camera_r",
}


def normalize_voyager_interaction(interaction):
    """Map official Voyager README action names to the runtime action keys."""
    if not isinstance(interaction, str):
        return interaction
    key = interaction.strip()
    return _INTERACTION_ALIASES.get(key, _INTERACTION_ALIASES.get(key.lower(), key))


def camera_list(
    num_frames=49,
    type="forward",
    Width=512,
    Height=512,
    fx=256,
    fy=256,
    prev_extrinsic=None,
):
    """
    Generate camera intrinsics and extrinsics.

    When prev_extrinsic is provided, the trajectory continues from the previous
    segment's last world-to-camera matrix.
    """
    type = normalize_voyager_interaction(type)
    cx = Width // 2
    cy = Height // 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics = np.stack([intrinsic] * num_frames)

    if prev_extrinsic is not None:
        R_prev = prev_extrinsic[:3, :3]
        start_pos = -R_prev.T @ prev_extrinsic[:3, 3]
        cam_right = R_prev[0, :].copy()
        cam_up = R_prev[1, :].copy()
        cam_forward = R_prev[2, :].copy()
    else:
        start_pos = np.array([0.0, 0.0, 0.0])
        cam_right = np.array([1.0, 0.0, 0.0])
        cam_up = np.array([0.0, 1.0, 0.0])
        cam_forward = np.array([0.0, 0.0, 1.0])

    if type == "forward":
        end_pos = start_pos + cam_forward
    elif type == "backward":
        end_pos = start_pos - cam_forward
    elif type == "left":
        end_pos = start_pos - cam_right
    elif type == "right":
        end_pos = start_pos + cam_right
    else:
        end_pos = start_pos.copy()

    camera_centers = np.linspace(start_pos, end_pos, num_frames)

    if type == "camera_l":
        target_start = start_pos + cam_forward * 100
        target_end = start_pos - cam_right * 100
        target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]
    elif type == "camera_r":
        target_start = start_pos + cam_forward * 100
        target_end = start_pos + cam_right * 100
        target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]
    else:
        target_points = camera_centers + cam_forward[np.newaxis, :] * 100

    extrinsics = []
    for t, target_point in zip(camera_centers, target_points):
        z = target_point - t
        z = z / np.linalg.norm(z)

        y = np.cross(z, cam_right)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-6:
            y = cam_up.copy()
            norm_y = np.linalg.norm(y)
        y = y / norm_y
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)

    extrinsics = np.stack(extrinsics)
    return intrinsics, extrinsics


class HunyuanWorldVoyagerOperator(BaseOperator):
    """Operator class for the HunyuanWorldVoyager model integration."""
    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["action_instruction"]
        if interaction_template is None:
            interaction_template = [
                "forward",
                "backward",
                "left",
                "right",
                "camera_l",
                "camera_r",
                "turn_left",
                "turn_right",
            ]
        super(HunyuanWorldVoyagerOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        normalized = normalize_voyager_interaction(interaction)
        if interaction not in self.interaction_template and normalized not in self.interaction_template:
            raise ValueError(f"Interaction {interaction} not in interaction_template")
        return True

    def get_interaction(self, interaction):
        """Support a single interaction or an interaction sequence."""
        if isinstance(interaction, str):
            if self.check_interaction(interaction):
                self.current_interaction.append(normalize_voyager_interaction(interaction))
        elif isinstance(interaction, list):
            for inter in interaction:
                if self.check_interaction(inter):
                    self.current_interaction.append(normalize_voyager_interaction(inter))
        else:
            raise ValueError(f"Interaction must be a string or list, got {type(interaction)}")

    def process_interaction(
        self,
        num_frames,
        Width=512,
        Height=512,
        fx=256,
        fy=256,
    ):
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return camera_list(
            num_frames=num_frames,
            type=now_interaction,
            Width=Width,
            Height=Height,
            fx=fx,
            fy=fy,
        )

    def process_interaction_sequence(
        self,
        interaction_sequence,
        num_frames_per_interaction,
        Width=512,
        Height=512,
        fx=256,
        fy=256,
    ):
        """Process an interaction sequence and return all camera parameters."""
        all_intrinsics = []
        all_extrinsics = []

        for interaction in interaction_sequence:
            if self.check_interaction(interaction):
                interaction = normalize_voyager_interaction(interaction)
                self.interaction_history.append(interaction)
                intrinsics, extrinsics = camera_list(
                    num_frames=num_frames_per_interaction,
                    type=interaction,
                    Width=Width,
                    Height=Height,
                    fx=fx,
                    fy=fy,
                )
                all_intrinsics.append(intrinsics)
                all_extrinsics.append(extrinsics)

        if len(all_intrinsics) > 0:
            all_intrinsics = np.concatenate(all_intrinsics, axis=0)
            all_extrinsics = np.concatenate(all_extrinsics, axis=0)
        else:
            raise ValueError("No valid interactions in sequence")

        return all_intrinsics, all_extrinsics

    def process_perception(self, input_image, device):
        """Process perception inputs like images, videos, and reference frames."""
        if isinstance(input_image, np.ndarray):
            image_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        elif isinstance(input_image, Image.Image):
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            input_image = np.array(input_image)
            image_tensor = torch.tensor(input_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        else:
            image_tensor = input_image.to(device)
        return input_image, image_tensor


__all__ = ["HunyuanWorldVoyagerOperator", "camera_list", "normalize_voyager_interaction"]
