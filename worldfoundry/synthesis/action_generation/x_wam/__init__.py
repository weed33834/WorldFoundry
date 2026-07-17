"""Inference-only X-WAM world-action synthesis package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "OFFICIAL_CHECKPOINT": (".inference", "OFFICIAL_CHECKPOINT"),
    "OFFICIAL_VARIANTS": (".inference", "OFFICIAL_VARIANTS"),
    "XWAMModel": (".modeling", "XWAMModel"),
    "XWAMRuntime": (".inference", "XWAMRuntime"),
    "XWAMRuntimeConfig": (".inference", "XWAMRuntimeConfig"),
    "XWAMSynthesis": (".x_wam_synthesis", "XWAMSynthesis"),
    "predict_action": (".runtime", "predict_action"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = sorted(_EXPORTS)
