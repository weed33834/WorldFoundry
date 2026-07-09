# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p2 -> __init__.py functionality."""

from importlib import import_module

__all__ = [
    "configs",
    "distributed",
    "modules",
    "WanAnimate",
    "WanI2V",
    "WanS2V",
    "WanT2V",
    "WanTI2V",
]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name in {"configs", "distributed", "modules"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    lazy_modules = {
        "WanAnimate": (".animate", "WanAnimate"),
        "WanI2V": (".image2video", "WanI2V"),
        "WanS2V": (".speech2video", "WanS2V"),
        "WanT2V": (".text2video", "WanT2V"),
        "WanTI2V": (".textimage2video", "WanTI2V"),
    }
    if name not in lazy_modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = lazy_modules[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
