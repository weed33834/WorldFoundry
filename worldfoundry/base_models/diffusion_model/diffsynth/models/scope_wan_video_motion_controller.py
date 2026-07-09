"""Module for base_models -> diffusion_model -> diffsynth -> models -> scope_wan_video_motion_controller.py functionality."""

import torch
import torch.nn as nn
from .scope_wan_video_dit import sinusoidal_embedding_1d



class WanMotionControllerModel(torch.nn.Module):
    """Wan motion controller model implementation."""
    def __init__(self, freq_dim=256, dim=1536):
        """Init.

        Args:
            freq_dim: The freq dim.
            dim: The dim.
        """
        super().__init__()
        self.freq_dim = freq_dim
        self.linear = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, motion_bucket_id):
        """Forward.

        Args:
            motion_bucket_id: The motion bucket id.
        """
        emb = sinusoidal_embedding_1d(self.freq_dim, motion_bucket_id * 10)
        emb = self.linear(emb)
        return emb

    def init(self):
        """Init."""
        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)
