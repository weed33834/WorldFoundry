"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p1 -> __init__.py functionality."""

from importlib import import_module

__all__ = [
    "WanFLF2V",
    "WanI2V",
    "WanT2V",
    "WanVace",
    "WanVaceMP",
    "configs",
    "distributed",
    "modules",
    "sp",
]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name in {"configs", "distributed", "modules", "sp"}:
        return import_module(f"{__name__}.{name}")
    if name == "WanFLF2V":
        from .first_last_frame2video import WanFLF2V

        return WanFLF2V
    if name == "WanI2V":
        from .image2video import WanI2V

        return WanI2V
    if name == "WanT2V":
        from .text2video import WanT2V

        return WanT2V
    if name in {"WanVace", "WanVaceMP"}:
        from .vace import WanVace, WanVaceMP

        return {"WanVace": WanVace, "WanVaceMP": WanVaceMP}[name]
    raise AttributeError(name)
