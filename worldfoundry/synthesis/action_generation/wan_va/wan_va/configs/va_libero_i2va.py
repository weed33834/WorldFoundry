# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .shared_config import _load_yaml_config
from .va_libero_cfg import va_libero_cfg

va_libero_i2va_cfg = EasyDict(__name__='Config: VA libero i2va')
va_libero_i2va_cfg.update(va_libero_cfg)
va_libero_i2va_cfg.update(_load_yaml_config("libero_i2va.yaml"))
