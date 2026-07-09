# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .shared_config import _load_yaml_config
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_cfg = EasyDict(__name__='Config: VA robotwin i2va')
va_robotwin_i2va_cfg.update(va_robotwin_cfg)
va_robotwin_i2va_cfg.update(_load_yaml_config("robotwin_i2va.yaml"))
