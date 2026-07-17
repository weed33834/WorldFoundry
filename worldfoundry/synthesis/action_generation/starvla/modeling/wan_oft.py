# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Inference-only Wan2.2 world-model OFT policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .action_head import build_action_head
from .base import StarVLAInferenceModel
from .config import add_discretized_state_to_instruction, merge_framework_config
from .images import to_pil_preserve
from .utils import resize_images
from .wan2 import Wan2Interface


@dataclass
class WanOFTDefaultConfig:
    name: str = "WanOFT"
    world_model: dict[str, Any] = field(
        default_factory=lambda: {
            "base_wm": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
        }
    )
    qwenvl: dict[str, Any] = field(default_factory=dict)
    action_model: dict[str, Any] = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 3072,
            "future_action_window_size": 7,
            "past_action_window_size": 0,
        }
    )


class WanOFT(StarVLAInferenceModel):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = merge_framework_config(WanOFTDefaultConfig, config)
        self.backbone = Wan2Interface(self.config)
        hidden_dim = int(self.backbone.hidden_size)
        self.config.framework.action_model.action_hidden_dim = hidden_dim
        self.action_model = build_action_head(self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_query_proj = nn.Linear(hidden_dim, self.chunk_len * hidden_dim)

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_dim = hidden_states.shape
        projection_dtype = self.action_query_proj.weight.dtype
        projected = self.action_query_proj(hidden_states.mean(dim=1).to(dtype=projection_dtype))
        return projected.view(batch_size, self.chunk_len, hidden_dim)

    @torch.inference_mode()
    def predict_action(self, examples: list[dict[str, Any]], **_: Any) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("StarVLA WanOFT requires at least one example.")

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [str(example["lang"]) for example in examples]
        has_state = [example.get("state") is not None for example in examples]
        if any(has_state) and not all(has_state):
            raise ValueError("StarVLA batches cannot mix examples with and without state.")
        if all(has_state):
            instructions = add_discretized_state_to_instruction(
                instructions, [example["state"] for example in examples]
            )

        datasets = getattr(self.config, "datasets", None)
        vla_data = getattr(datasets, "vla_data", None) if datasets is not None else None
        image_size = vla_data.get("obs_image_size", None) if vla_data is not None else None
        if image_size:
            batch_images = resize_images(batch_images, target_size=int(image_size))

        outputs = self.backbone(**self.backbone.build_inputs(batch_images, instructions))
        action_queries = self._pool_to_action_queries(outputs.hidden_states[-1])
        head_dtype = next(self.action_model.parameters()).dtype
        predicted = self.action_model.predict_action(action_queries.to(dtype=head_dtype))
        return {"normalized_actions": predicted.float().cpu().numpy()}


__all__ = ["WanOFT"]
