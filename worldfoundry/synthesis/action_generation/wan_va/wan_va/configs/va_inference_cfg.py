# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import _inverse_action_channels, _load_yaml_config, va_shared_cfg


va_inference_cfg = EasyDict(__name__='Config: VA inference')
va_inference_cfg.update(va_shared_cfg)

va_inference_cfg.update(_load_yaml_config("demo.yaml"))
va_inference_cfg.inverse_used_action_channel_ids = _inverse_action_channels(
    va_inference_cfg.action_dim,
    list(va_inference_cfg.used_action_channel_ids),
)
