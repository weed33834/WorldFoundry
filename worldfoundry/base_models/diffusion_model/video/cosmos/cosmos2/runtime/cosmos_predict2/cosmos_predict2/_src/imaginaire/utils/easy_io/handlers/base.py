# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> utils -> easy_io -> handlers -> base.py functionality."""

from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    """Base file handler implementation."""
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        """Load from fileobj.

        Args:
            file: The file.
        """
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to fileobj.

        Args:
            obj: The obj.
            file: The file.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        """Dump to str.

        Args:
            obj: The obj.
        """
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        """Load from path.

        Args:
            filepath: The filepath.
            mode: The mode.
        """
        with open(filepath, mode) as handle:
            return self.load_from_fileobj(handle, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        """Dump to path.

        Args:
            obj: The obj.
            filepath: The filepath.
            mode: The mode.
        """
        with open(filepath, mode) as handle:
            self.dump_to_fileobj(obj, handle, **kwargs)
