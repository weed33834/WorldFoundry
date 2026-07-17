"""Inference-only in-tree X-VLA integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "XVLA": ("worldfoundry.synthesis.action_generation.xvla.modeling", "XVLA"),
    "XVLAConfig": ("worldfoundry.synthesis.action_generation.xvla.configuration", "XVLAConfig"),
    "XVLAProcessor": ("worldfoundry.synthesis.action_generation.xvla.processing", "XVLAProcessor"),
    "XVLARuntime": ("worldfoundry.synthesis.action_generation.xvla.runtime", "XVLARuntime"),
    "XVLARuntimeConfig": ("worldfoundry.synthesis.action_generation.xvla.runtime", "XVLARuntimeConfig"),
    "XVLASynthesis": ("worldfoundry.synthesis.action_generation.xvla.xvla_synthesis", "XVLASynthesis"),
    "predict_action": ("worldfoundry.synthesis.action_generation.xvla.runtime", "predict_action"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = sorted(_EXPORTS)
