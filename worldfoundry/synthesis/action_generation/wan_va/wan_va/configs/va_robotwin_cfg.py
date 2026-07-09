# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import _inverse_action_channels, _load_yaml_config, va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)
va_robotwin_cfg.update(_load_yaml_config("robotwin.yaml"))
va_robotwin_cfg.inverse_used_action_channel_ids = _inverse_action_channels(
    va_robotwin_cfg.action_dim,
    list(va_robotwin_cfg.used_action_channel_ids),
)
