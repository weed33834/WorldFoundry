# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_inference_cfg import _load_yaml_config, va_inference_cfg

va_inference_i2va_cfg = EasyDict(__name__='Config: VA inference i2va')
va_inference_i2va_cfg.update(va_inference_cfg)
va_inference_i2va_cfg.update(_load_yaml_config("demo_i2va.yaml"))
