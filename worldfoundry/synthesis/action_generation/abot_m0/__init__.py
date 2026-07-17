"""ABot-M0 in-tree inference integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ABotM0Synthesis": ".abot_m0_synthesis",
    "ABotM0Runtime": ".runtime",
    "ABotM0RuntimeConfig": ".runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
