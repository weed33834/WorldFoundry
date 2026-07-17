"""Inference-only in-tree Spirit-v1.5 integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "SpiritVLAPolicy": ("worldfoundry.synthesis.action_generation.spirit_v15.modeling", "SpiritVLAPolicy"),
    "SpiritV15Runtime": ("worldfoundry.synthesis.action_generation.spirit_v15.runtime", "SpiritV15Runtime"),
    "SpiritV15RuntimeConfig": (
        "worldfoundry.synthesis.action_generation.spirit_v15.runtime",
        "SpiritV15RuntimeConfig",
    ),
    "SpiritV15Synthesis": (
        "worldfoundry.synthesis.action_generation.spirit_v15.spirit_v15_synthesis",
        "SpiritV15Synthesis",
    ),
    "predict_action": ("worldfoundry.synthesis.action_generation.spirit_v15.runtime", "predict_action"),
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
