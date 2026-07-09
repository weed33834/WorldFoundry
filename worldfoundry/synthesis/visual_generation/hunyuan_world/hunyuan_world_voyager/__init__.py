from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HunyuanWorldVoyagerRuntime",
    "get_1d_rotary_pos_embed_riflex",
    "load_models",
    "load_runtime",
    "parallelize_transformer",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".runtime", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
