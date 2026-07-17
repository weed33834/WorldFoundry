"""Inference-only X-VLA action-space transforms.

Adapted from ``models/action_hub.py`` in 2toINF/X-VLA at revision
``6bc2513f5f1cbec715cc668b414392a6cae5c671``.  Supervised losses and
training-only registry hooks are intentionally omitted.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ActionSpace(nn.Module):
    """Base inference transform for an X-VLA action representation."""

    dim_action = 0
    dim_proprio = 0

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action


class _GripperActionSpace(ActionSpace):
    gripper_idx: tuple[int, ...] = ()

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = proprio.clone()
        action = action.clone()
        proprio[..., self.gripper_idx] = 0.0
        action[..., self.gripper_idx] = 0.0
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clone()
        action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


class EE6DActionSpace(_GripperActionSpace):
    """Two-arm EE6D layout: xyz + rotation-6D + gripper per arm."""

    dim_action = 20
    dim_proprio = 20
    gripper_idx = (9, 19)


class JointActionSpace(_GripperActionSpace):
    """Two-arm joint-space layout with one gripper channel per arm."""

    dim_action = 14
    dim_proprio = 14
    gripper_idx = (6, 13)


class AgibotEE6DActionSpace(ActionSpace):
    """AGIBOT EE6D layout, which requires no inference transform."""

    dim_action = 20
    dim_proprio = 20


class AutoActionSpace(ActionSpace):
    """Pad an arbitrary real action width to the pretrained model width."""

    def __init__(self, *, real_dim: int, max_dim: int = 20) -> None:
        super().__init__()
        if real_dim <= 0 or max_dim <= 0 or real_dim > max_dim:
            raise ValueError(f"expected 0 < real_dim <= max_dim, got {real_dim=} {max_dim=}")
        self.real_dim = int(real_dim)
        self.dim_action = int(max_dim)
        # Upstream checkpoints always build the proprio projection at the
        # model-facing width. ``real_dim`` only controls the returned action
        # width; changing this to ``real_dim`` changes checkpoint tensor shapes.
        self.dim_proprio = int(max_dim)

    def _pad(self, value: torch.Tensor) -> torch.Tensor:
        if int(value.shape[-1]) == self.dim_action:
            return value
        value = value[..., : self.real_dim]
        if int(value.shape[-1]) < self.real_dim:
            value = torch.nn.functional.pad(value, (0, self.real_dim - int(value.shape[-1])))
        return torch.nn.functional.pad(value, (0, self.dim_action - self.real_dim))

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return proprio, self._pad(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., : self.real_dim]


def build_action_space(name: str, **kwargs: Any) -> ActionSpace:
    """Build one of the action layouts released by X-VLA."""

    key = str(name).strip().lower().replace("-", "_")
    action_spaces: dict[str, type[ActionSpace]] = {
        "ee6d": EE6DActionSpace,
        "joint": JointActionSpace,
        "agibot_ee6d": AgibotEE6DActionSpace,
        "auto": AutoActionSpace,
    }
    try:
        action_space = action_spaces[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported X-VLA action mode {name!r}; expected {sorted(action_spaces)}") from exc
    return action_space(**kwargs)


__all__ = [
    "ActionSpace",
    "AgibotEE6DActionSpace",
    "AutoActionSpace",
    "EE6DActionSpace",
    "JointActionSpace",
    "build_action_space",
]
