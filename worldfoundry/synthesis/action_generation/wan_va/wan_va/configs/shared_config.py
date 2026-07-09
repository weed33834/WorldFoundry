# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from pathlib import Path

import torch
import yaml

from easydict import EasyDict

WORLDFOUNDRY_DATA_ROOT = Path(__file__).resolve().parents[5] / "data"
WAN_VA_RUNTIME_CONFIG_ROOT = WORLDFOUNDRY_DATA_ROOT / "models" / "runtime" / "configs" / "wan_va"


def _load_yaml_config(name: str) -> dict:
    path = WAN_VA_RUNTIME_CONFIG_ROOT / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def _inverse_action_channels(action_dim: int, used_action_channel_ids: list[int]) -> list[int]:
    inverse = [len(used_action_channel_ids)] * action_dim
    for index, channel_id in enumerate(used_action_channel_ids):
        inverse[channel_id] = index
    return inverse


def _torch_dtype(name: str):
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype in Wan-VA YAML: {name}") from exc


va_shared_cfg = EasyDict(__name__="Config: VA shared")
_shared_payload = _load_yaml_config("shared.yaml")
_dtype_name = _shared_payload.pop("param_dtype", "bfloat16")
va_shared_cfg.update(_shared_payload)
va_shared_cfg.param_dtype = _torch_dtype(_dtype_name)
