"""Small lazy-module helper for optional runtime dependencies."""

from __future__ import annotations

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any


class LazyModule(ModuleType):
    """Module wrapper that imports submodules only when attributes are requested."""

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        module_spec: Any | None = None,
        extra_objects: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> list[str]:
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self) -> tuple[type["LazyModule"], tuple[str, str, dict[str, list[str]]]]:
        return self.__class__, (self._name, self.__file__, self._import_structure)


_LazyModule = LazyModule


__all__ = ["LazyModule", "_LazyModule"]
