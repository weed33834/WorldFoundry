"""Checkpoint loading and state-dict remapping helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "get_storage_reader": "worldfoundry.core.checkpoint.load",
    "load_checkpoint": "worldfoundry.core.checkpoint.load",
    "load_distributed_checkpoint": "worldfoundry.core.checkpoint.load",
    "load_sharded_safetensors_parallel_with_progress": "worldfoundry.core.checkpoint.sharded_safetensors",
    "load_single_checkpoint": "worldfoundry.core.checkpoint.load",
    "remap_checkpoint_keys": "worldfoundry.core.checkpoint.remap",
    "unwrap_model": "worldfoundry.core.checkpoint.sharded_safetensors",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
