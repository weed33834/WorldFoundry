"""Module for base_models -> diffusion_model -> diffsynth -> extensions -> FastBlend -> __init__.py functionality."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "FastBlendSmoother": "worldfoundry.base_models.diffusion_model.diffsynth.processors.FastBlend",
    "PyramidPatchMatcher": (
        "worldfoundry.base_models.diffusion_model.diffsynth.extensions.FastBlend.patch_match"
    ),
    "TableManager": (
        "worldfoundry.base_models.diffusion_model.diffsynth.extensions.FastBlend.runners.fast"
    ),
}


def __getattr__(name: str) -> Any:
    """Getattr.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Dir.

    Returns:
        The return value.
    """
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
