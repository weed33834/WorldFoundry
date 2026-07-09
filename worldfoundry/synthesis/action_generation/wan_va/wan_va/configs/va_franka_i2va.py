# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .shared_config import _load_yaml_config
from .va_franka_cfg import va_franka_cfg

va_franka_i2va_cfg = EasyDict(__name__='Config: VA franka i2va')
va_franka_i2va_cfg.update(va_franka_cfg)
va_franka_i2va_cfg.update(_load_yaml_config("franka_i2va.yaml"))
