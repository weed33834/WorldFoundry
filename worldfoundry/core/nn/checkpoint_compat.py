"""Neural-network compatibility helpers required when loading released checkpoints."""

from __future__ import annotations

import torch
from torch import nn


class InferenceCheckpointModule(nn.Module):
    """Preserve legacy checkpoint metadata buffers without training utilities."""

    def __init__(self) -> None:
        super().__init__()
        buffers = {
            "accum_video_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_image_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_iteration": torch.tensor(0, dtype=torch.int64),
            "accum_train_in_hours": torch.tensor(0.0, dtype=torch.float32),
        }
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)


__all__ = ["InferenceCheckpointModule"]
