"""Minimal lazy-config target resolution helpers for in-tree inference."""

from __future__ import annotations

import importlib
import pydoc
from typing import Any


def _convert_target_to_string(target: Any) -> str:
    """Return an importable string for a callable/class target."""

    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        raise TypeError(f"Cannot convert target to import string: {target!r}")
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """Locate an object by dotted import path."""

    found = pydoc.locate(name)
    if found is not None:
        return found

    parts = name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        attr_parts = parts[index:]
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in attr_parts:
            obj = getattr(obj, attr)
        return obj
    return None
