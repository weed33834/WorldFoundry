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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> utils -> easy_io -> handlers -> tarfile_handler.py functionality."""

import tarfile

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TarHandler(BaseFileHandler):
    """Tar handler implementation."""
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        """Load from fileobj.

        Args:
            file: The file.
            mode: The mode.
        """
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        """Load from path.

        Args:
            filepath: The filepath.
            mode: The mode.
        """
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        """Dump to fileobj.

        Args:
            obj: The obj.
            file: The file.
            mode: The mode.
        """
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        """Dump to path.

        Args:
            obj: The obj.
            filepath: The filepath.
            mode: The mode.
        """
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Dump to str.

        Args:
            obj: The obj.
        """
        raise NotImplementedError
