from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Dynamics(nn.Module):
    def __init__(self, action_dim: int, action_num: int, hidden_size: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size

        self.joint_vel_01 = np.array(
            [-0.4077107, -0.79047304, -0.47850373, -0.8666644, -0.6729502, -0.5602032, -0.692411]
        )[None, :]
        self.joint_vel_99 = np.array(
            [0.4900636, 0.7259861, 0.45910007, 0.79220384, 0.69864315, 0.648198, 0.810115]
        )[None, :]
        self.joint_delta_01 = np.array(
            [-0.2801219, -0.397792, -0.22935797, -0.3351759, -0.42025003, -0.36825255, -0.450706]
        )[None, :]
        self.joint_delta_99 = np.array(
            [0.2827909, 0.42184818, 0.33529875, 0.35958457, 0.375613, 0.44463825, 0.4697690]
        )[None, :]

        input_dim = int(action_dim * (action_num + 1))
        output_dim = int(action_num * action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        joint: np.ndarray,
        joint_vel: np.ndarray,
        joint_delta: np.ndarray | None,
        training: bool = True,
    ) -> torch.Tensor | np.ndarray:
        if joint.ndim == 2:
            joint = joint[None, :]
        if joint_vel.ndim == 2:
            joint_vel = joint_vel[None, :]
        assert joint.shape[1:] == (1, self.action_dim), (
            f"Joint shape should be (B, 1, action_dim), got {joint.shape}"
        )
        assert joint_vel.shape[1:] == (self.action_num, self.action_dim), (
            f"Joint velocity shape should be (B, action_num, action_dim), got {joint_vel.shape}"
        )

        joint_tensor = torch.tensor(joint).float().to(self.device)
        joint_vel = self.normalize_bound(joint_vel, self.joint_vel_01, self.joint_vel_99)
        joint_vel_tensor = torch.tensor(joint_vel).float().to(self.device)

        batch_size = joint_tensor.shape[0]
        joint_tensor = joint_tensor.reshape(batch_size, -1)
        joint_vel_tensor = joint_vel_tensor.reshape(batch_size, -1)
        model_input = torch.cat((joint_tensor, joint_vel_tensor), dim=1)
        pred = self.net(model_input)
        pred = pred.reshape(batch_size, self.action_num, self.action_dim)

        if training:
            if joint_delta is None:
                raise ValueError("joint_delta is required when training=True")
            joint_delta = self.normalize_bound(joint_delta, self.joint_delta_01, self.joint_delta_99)
            joint_delta_tensor = torch.tensor(joint_delta).float().to(self.device)
            return F.mse_loss(pred, joint_delta_tensor)

        pred = pred.detach().cpu().numpy()
        pred = self.denormalize_bound(pred, self.joint_delta_01, self.joint_delta_99)
        joint_array = joint_tensor.detach().cpu().numpy().reshape(batch_size, 1, self.action_dim)
        joint_future = joint_array + pred
        return joint_future[0]

    @staticmethod
    def normalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        del clip_min, clip_max
        return 2 * (data - data_min) / (data_max - data_min + eps) - 1

    @staticmethod
    def denormalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        return (data - clip_min) / clip_range * (data_max - data_min) + data_min
