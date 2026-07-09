# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-16 16:03:17
# @Last Modified by:   Shilong Liu
# @Last Modified time: 2022-01-23 15:26
# modified from mmcv

"""Module for base_models -> perception_core -> detection -> grounding_dino -> registry_core.py functionality."""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable


class Registry:
    """Registry implementation."""
    def __init__(self, name: str) -> None:
        """Init.

        Args:
            name: The name.

        Returns:
            The return value.
        """
        self._name = name
        self._module_dict: dict[str, Callable[..., Any]] = {}

    def __repr__(self) -> str:
        """Repr.

        Returns:
            The return value.
        """
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})"

    def __len__(self) -> int:
        """Len.

        Returns:
            The return value.
        """
        return len(self._module_dict)

    @property
    def name(self) -> str:
        """Name.

        Returns:
            The return value.
        """
        return self._name

    @property
    def module_dict(self) -> dict[str, Callable[..., Any]]:
        """Module dict.

        Returns:
            The return value.
        """
        return self._module_dict

    def get(self, key: str) -> Callable[..., Any] | None:
        """Get.

        Args:
            key: The key.

        Returns:
            The return value.
        """
        return self._module_dict.get(key, None)

    def registe_with_name(self, module_name: str | None = None, force: bool = False) -> Any:
        """Registe with name.

        Args:
            module_name: The module name.
            force: The force.

        Returns:
            The return value.
        """
        return partial(self.register, module_name=module_name, force=force)

    def register(
        self,
        module_build_function: Callable[..., Any],
        module_name: str | None = None,
        force: bool = False,
    ) -> Callable[..., Any]:
        """Register.

        Args:
            module_build_function: The module build function.
            module_name: The module name.
            force: The force.

        Returns:
            The return value.
        """
        if not inspect.isfunction(module_build_function):
            raise TypeError(f"module_build_function must be a function, but got {type(module_build_function)}")
        resolved_name = module_name or module_build_function.__name__
        if not force and resolved_name in self._module_dict:
            raise KeyError(f"{resolved_name} is already registered in {self.name}")
        self._module_dict[resolved_name] = module_build_function
        return module_build_function


__all__ = ["Registry"]
