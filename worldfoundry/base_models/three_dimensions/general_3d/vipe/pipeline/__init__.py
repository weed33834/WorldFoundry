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


"""Module for base_models -> three_dimensions -> general_3d -> vipe -> pipeline -> __init__.py functionality."""

import copy
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, cast

from omegaconf import DictConfig

from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import BaseConfigSchema
from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.base import MultiviewVideoList, VideoStream


@dataclass(kw_only=True, slots=True)
class AnnotationPipelineOutput:
    """Annotation pipeline output implementation."""
    # Eager return of the payload values that comes from early stages of the pipeline.
    payload: Any | None = None
    output_streams: Sequence[VideoStream] | None = None


class Pipeline(ABC):
    """Pipeline implementation."""
    def __init__(self) -> None:
        """Init.

        Returns:
            The return value.
        """
        self._return_payload = False
        self._return_output_streams = False

    @property
    def return_payload(self) -> bool:
        """Return payload.

        Returns:
            The return value.
        """
        return self._return_payload

    @return_payload.setter
    def return_payload(self, value: bool) -> None:
        """Return payload.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        assert isinstance(value, bool), "return_payload must be a boolean"
        self._return_payload = value
        if value:
            self._return_output_streams = False

    @property
    def return_output_streams(self) -> bool:
        """Return output streams.

        Returns:
            The return value.
        """
        return self._return_output_streams

    @return_output_streams.setter
    def return_output_streams(self, value: bool) -> None:
        """Return output streams.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        assert isinstance(value, bool), "return_output_streams must be a boolean"
        self._return_output_streams = value
        if value:
            self._return_payload = False

    def should_filter(self, stream_name: str) -> bool:
        """Should filter.

        Args:
            stream_name: The stream name.

        Returns:
            The return value.
        """
        return False

    @abstractmethod
    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput: ...


def _as_dictconfig(config: DictConfig | BaseConfigSchema) -> DictConfig:
    """Helper function to as dictconfig.

    Args:
        config: The config.

    Returns:
        The return value.
    """
    if isinstance(config, BaseConfigSchema):
        return config.to_dictconfig()
    return config


def make_pipeline_cls(config: DictConfig | BaseConfigSchema) -> type[Pipeline]:
    """Make pipeline cls.

    Args:
        config: The config.

    Returns:
        The return value.
    """
    config = _as_dictconfig(config)
    module_path, class_name = config.instance.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_pipeline(config: DictConfig | BaseConfigSchema) -> Pipeline:
    """Make pipeline.

    Args:
        config: The config.

    Returns:
        The return value.
    """
    config = copy.deepcopy(_as_dictconfig(config))
    pipeline_cls = make_pipeline_cls(config)
    del config.instance
    return pipeline_cls(**cast(dict[str, Any], config))
