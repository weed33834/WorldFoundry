# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> three_dimensions -> depth -> depth_anything -> depth_anything_v3 -> utils -> registry.py functionality."""

from typing import Any

from addict import Dict as AttrDict


class Registry(dict[str, Any]):
    """Registry implementation."""
    def __init__(self):
        """Init."""
        super().__init__()
        self._map = AttrDict({})

    def register(self, name=None):
        """Register.

        Args:
            name: The name.
        """
        def decorator(cls):
            """Decorator."""
            key = name or cls.__name__
            self._map[key] = cls
            return cls

        return decorator

    def get(self, name):
        """Get.

        Args:
            name: The name.
        """
        return self._map[name]

    def all(self):
        """All."""
        return self._map
