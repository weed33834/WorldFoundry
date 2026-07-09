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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> utils -> easy_io -> handlers -> pandas_handler.py functionality."""

import pandas as pd

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler  # isort:skip


class PandasHandler(BaseFileHandler):
    """Pandas handler implementation."""
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        """Load from fileobj.

        Args:
            file: The file.
        """
        return pd.read_csv(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to fileobj.

        Args:
            obj: The obj.
            file: The file.
        """
        obj.to_csv(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Dump to str.

        Args:
            obj: The obj.
        """
        raise NotImplementedError("PandasHandler does not support dumping to str")
