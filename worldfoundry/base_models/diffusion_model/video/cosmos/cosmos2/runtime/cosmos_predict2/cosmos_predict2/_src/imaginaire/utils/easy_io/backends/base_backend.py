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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> utils -> easy_io -> backends -> base_backend.py functionality."""

import io
import os
import os.path as osp
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union


def mkdir_or_exist(dir_name, mode=0o777):
    """Mkdir or exist.

    Args:
        dir_name: The dir name.
        mode: The mode.
    """
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def has_method(obj, method):
    """Has method.

    Args:
        obj: The obj.
        method: The method.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends."""

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink: bool = False

    @property
    def allow_symlink(self) -> bool:
        """Allow symlink.

        Returns:
            The return value.
        """
        return self._allow_symlink

    @property
    def name(self) -> str:
        """Name.

        Returns:
            The return value.
        """
        return self.__class__.__name__

    @abstractmethod
    def size(self, filepath: Union[str, Path]) -> int:
        """Size.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        """Get.

        Args:
            filepath: The filepath.
            offset: The offset.
            size: The size.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Get text.

        Args:
            filepath: The filepath.
            encoding: The encoding.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        """Put.

        Args:
            obj: The obj.
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def put_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        """Put text.

        Args:
            obj: The obj.
            filepath: The filepath.
            encoding: The encoding.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def exists(self, filepath: Union[str, Path]) -> bool:
        """Exists.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Isdir.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Isfile.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        """Join path.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        """Get local path.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copyfile(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        """Copyfile.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copytree(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        """Copytree.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copyfile_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        """Copyfile from local.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copytree_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        """Copytree from local.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dst_type: str,  # Choose from ["file", "dir"]
    ) -> Union[str, Path]:
        """Copyfile to local.

        Args:
            src: The src.
            dst: The dst.
            dst_type: The dst type.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copytree_to_local(self, src: Union[str, Path], dst: Union[str, Path]) -> Union[str, Path]:
        """Copytree to local.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove.

        Args:
            filepath: The filepath.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Rmtree.

        Args:
            dir_path: The dir path.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def copy_if_symlink_fails(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy if symlink fails.

        Args:
            src: The src.
            dst: The dst.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def list_dir(self, dir_path: Union[str, Path]) -> Generator[str, None, None]:
        """List dir.

        Args:
            dir_path: The dir path.

        Returns:
            The return value.
        """
        pass

    @abstractmethod
    def list_dir_or_file(  # pylint: disable=too-many-arguments
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """List dir or file.

        Args:
            dir_path: The dir path.
            list_dir: The list dir.
            list_file: The list file.
            suffix: The suffix.
            recursive: The recursive.

        Returns:
            The return value.
        """
        pass
