"""Lightweight Python config loading helpers."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


class EasyDict(dict):
    """Dictionary with recursive attribute access."""

    def __init__(self, mapping: Any | None = None, **kwargs: Any) -> None:
        super().__init__()
        mapping = {} if mapping is None else mapping
        items = mapping.items() if hasattr(mapping, "items") else mapping
        for key, value in items:
            self[key] = self._wrap(value)
        for key, value in kwargs.items():
            self[key] = self._wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


def load_python_config(config_file: str | Path, overrides: Iterable[Any] = ()) -> EasyDict:
    cfg = EasyDict(_module_public_values(Path(config_file)))
    merge_list(cfg, list(overrides))
    eval_leaf_values(cfg)
    return cfg


def merge_list(cfg: EasyDict, opts: list[Any]) -> EasyDict:
    if len(opts) % 2 != 0:
        raise ValueError(f"Expected key/value override pairs, got {opts!r}")
    for idx in range(0, len(opts), 2):
        keys = str(opts[idx]).split(".")
        value = opts[idx + 1]
        node: Any = cfg
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"Config key does not exist: {'.'.join(keys)}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"Config key does not exist: {'.'.join(keys)}")
        node[keys[-1]] = EasyDict._wrap(value)
    return cfg


def eval_leaf_values(node: EasyDict, root: EasyDict | None = None) -> EasyDict:
    root = node if root is None else root
    for key, value in list(node.items()):
        if isinstance(value, EasyDict):
            eval_leaf_values(value, root)
        else:
            node[key] = EasyDict._wrap(_eval_string(value, root))
    return node


def _module_public_values(path: Path) -> dict[str, Any]:
    module_name = f"_worldfoundry_config_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return {
            name: deepcopy(value)
            for name, value in vars(module).items()
            if not name.startswith("__") and not isinstance(value, ModuleType)
        }
    finally:
        sys.modules.pop(module_name, None)


def _eval_string(value: Any, root: EasyDict) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("eval(") and value.endswith(")"):
        return eval(value[5:-1], {}, {"d": root})

    original = value
    resolved = re.sub(r"\${(.*)}", r"d.\1", original)
    while resolved != original:
        original = resolved
        resolved = re.sub(r"\${(.*)}", r"d.\1", original)
    if resolved != value:
        return eval(resolved, {}, {"d": root})

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


__all__ = ["EasyDict", "eval_leaf_values", "load_python_config", "merge_list"]
