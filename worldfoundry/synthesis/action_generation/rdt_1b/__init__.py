"""Inference-only in-tree RDT-1B integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RDT": ("worldfoundry.synthesis.action_generation.rdt_1b.modeling", "RDT"),
    "RDTRunner": ("worldfoundry.synthesis.action_generation.rdt_1b.runner", "RDTRunner"),
    "RDT1BRuntime": ("worldfoundry.synthesis.action_generation.rdt_1b.runtime", "RDT1BRuntime"),
    "RDT1BRuntimeConfig": (
        "worldfoundry.synthesis.action_generation.rdt_1b.runtime",
        "RDT1BRuntimeConfig",
    ),
    "RDT1BSynthesis": (
        "worldfoundry.synthesis.action_generation.rdt_1b.rdt_1b_synthesis",
        "RDT1BSynthesis",
    ),
    "predict_action": ("worldfoundry.synthesis.action_generation.rdt_1b.runtime", "predict_action"),
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
