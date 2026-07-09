# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_inference_cfg import va_inference_cfg
from .va_inference_i2va import va_inference_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_i2va import va_libero_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'demo': va_inference_cfg,
    'demo_i2av': va_inference_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_i2av': va_libero_i2va_cfg,
}
