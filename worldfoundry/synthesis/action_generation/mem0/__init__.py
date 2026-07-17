"""Inference-only in-tree Mem-0 integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Mem0Policy": ("worldfoundry.synthesis.action_generation.mem0.modeling", "Mem0Policy"),
    "Mem0Runtime": ("worldfoundry.synthesis.action_generation.mem0.runtime", "Mem0Runtime"),
    "Mem0RuntimeConfig": (
        "worldfoundry.synthesis.action_generation.mem0.runtime",
        "Mem0RuntimeConfig",
    ),
    "Mem0Synthesis": (
        "worldfoundry.synthesis.action_generation.mem0.mem0_synthesis",
        "Mem0Synthesis",
    ),
    "predict_action": ("worldfoundry.synthesis.action_generation.mem0.runtime", "predict_action"),
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
