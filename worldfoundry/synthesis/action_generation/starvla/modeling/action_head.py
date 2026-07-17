# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""MLP regression head used by the released StarVLA OFT checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.ffn(inputs)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(MLPResNetBlock(hidden_dim) for _ in range(num_blocks))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.fc1(self.layer_norm1(inputs)))
        for block in self.mlp_resnet_blocks:
            hidden = block(hidden)
        return self.fc2(self.layer_norm2(hidden))


class L1RegressionActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        NUM_ACTIONS_CHUNK: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.model = MLPResNet(2, input_dim, hidden_dim, action_dim)

    def predict_action(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, chunk_len, hidden_dim = action_hidden_states.shape
        actions = self.model(action_hidden_states.reshape(batch_size * chunk_len, hidden_dim))
        return actions.view(batch_size, chunk_len, self.action_dim)

    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(action_hidden_states)


def build_action_head(config) -> L1RegressionActionHead:
    action_config = config.framework.action_model
    return L1RegressionActionHead(
        input_dim=int(action_config.action_hidden_dim),
        hidden_dim=int(action_config.action_hidden_dim) * 2,
        action_dim=int(action_config.action_dim),
        NUM_ACTIONS_CHUNK=int(action_config.action_horizon),
    )


__all__ = ["L1RegressionActionHead", "build_action_head"]
