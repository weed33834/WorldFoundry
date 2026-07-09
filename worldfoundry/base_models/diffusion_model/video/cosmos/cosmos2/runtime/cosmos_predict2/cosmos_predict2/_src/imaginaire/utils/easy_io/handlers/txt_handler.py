# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> utils -> easy_io -> handlers -> txt_handler.py functionality."""

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TxtHandler(BaseFileHandler):
    """Txt handler implementation."""
    def load_from_fileobj(self, file, **kwargs):
        """Load from fileobj.

        Args:
            file: The file.
        """
        del kwargs
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to fileobj.

        Args:
            obj: The obj.
            file: The file.
        """
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        """Dump to str.

        Args:
            obj: The obj.
        """
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        return obj
