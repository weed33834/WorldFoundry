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

"""Runtime config system for the Lyra-2 inference package."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional, TypeVar, Union

import attrs
import torch

try:
    from megatron.core import ModelParallelConfig

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")

from lyra_2._ext.imaginaire.lazy_config import LazyDict
from worldfoundry.core.distributed import torch_process_group as distributed
from lyra_2._ext.imaginaire.utils.misc import Color

T = TypeVar("T")


def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls: T) -> T:
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)  # type: ignore

    cls.__setattr__ = setattr_override  # type: ignore

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True  # type: ignore

    cls.freeze = freeze  # type: ignore

    return cls


def _pretty_print_attrs_instance(obj: object, indent: int = 0, use_color: bool = False) -> str:
    """
    Recursively pretty prints attrs objects with color.
    """

    assert attrs.has(obj.__class__)

    lines: list[str] = []
    for attribute in attrs.fields(obj.__class__):
        value = getattr(obj, attribute.name)
        if attrs.has(value.__class__):
            if use_color:
                lines.append("   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ":")
            else:
                lines.append("   " * indent + "* " + attribute.name + ":")
            lines.append(_pretty_print_attrs_instance(value, indent + 1, use_color))
        else:
            if use_color:
                lines.append(
                    "   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ": " + Color.yellow(value)
                )
            else:
                lines.append("   " * indent + "* " + attribute.name + ": " + str(value))
    return "\n".join(lines)


def pretty_print_overrides(overrides: Optional[list[str]] = None, use_color: bool = False) -> str:
    """
    Pretty prints overrides.
    """

    lines: list[str] = []
    lines.append(Color.cyan("* ") + Color.green("overrides") + ": ")
    for override in overrides:
        if override == "--":
            continue
        if override.startswith("~"):
            attribute_name = override[1:]
            attribute_value = None
        else:
            attribute_name, attribute_value = override.split("=")
        if use_color:
            lines.append("   " + Color.cyan("* ") + Color.green(attribute_name) + ": " + Color.yellow(attribute_value))
        else:
            lines.append("   " + "* " + attribute_name + ": " + str(attribute_value))

    return "\n".join(lines)


@make_freezable
@attrs.define(slots=False)  # slots=False is required for make_freezable. See the make_freezable notes for more info.
class ObjectStoreConfig:
    # Whether the file I/O is from object store instead of local disk.
    enabled: bool = False
    # Path to the object store credentials file.
    credentials: str = ""
    # Object store bucket to read from / write to the objects.
    bucket: str = ""


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    # Project name.
    project: str = ""
    # Experiment name.
    group: str = ""
    # Run/job name.
    name: str = ""
    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", os.path.join(tempfile.gettempdir(), "imaginaire4-output"))
        return f"{local_root}/{self.path}"


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EMA decay rate.
    beta: float = 0.9999
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class PowerEMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EDM2 paper EMA decay rate.
    s: float = 0.1
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    # Set to True for better reproducibility of the results (only using deterministic cudnn functions).
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class JITConfig:
    # Enable exporting a JIT compiled model.
    enabled: bool = False
    # Input tensor shape, for example input.
    input_shape: Union[list[int], None] = None
    # Device to compile onto.
    device: str = "cuda"
    # # Data type to compile onto.
    dtype: str = "bfloat16"
    # Strict mode for PyTorch JIT.
    strict: bool = True


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    # possible checkpoint class
    type: Optional[Dict] = None
    # for dcp, whether to use async mode
    dcp_async_mode_enabled: bool = False
    # Configs for saving the checkpoints to object store.
    save_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    # Save the checkpoint every N iterations.
    save_iter: int = 999999999
    # Configs for loading the checkpoints from object store.
    load_from_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    # Path of model weights to resume the checkpoint from.
    load_path: str = ""
    # Load state_dict to the models in strict mode.
    strict_resume: bool = True
    # Configs for JIT compiling EMA model.
    jit: JITConfig = attrs.field(factory=JITConfig)
    # Print detailed information during checkpoint saving/loading.
    verbose: bool = True
    # keys not to resume from the checkpoint.
    keys_not_to_resume: list[str] = []
    # Whether to use the local filesystem for broadcasting checkpoint data (used for Tensor Parallel Checkpointer).
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False  # used in inference, load EMA weights to regular model
    # In dcp planner, skip the weight shape check, load weights into the model even weight shape is different
    dcp_allow_mismatched_size: bool = False


@make_freezable
@attrs.define(slots=False)
class RuntimeConfig:
    # cuDNN configs.
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    # Set the random seed for deterministic model loading.
    seed: int = 0
    memory_format: torch.memory_format = torch.preserve_format


@make_freezable
@attrs.define(slots=False)
class Config:
    """Config for the Lyra-2 inference runtime."""

    # Model configs.
    model: LazyDict
    # Runtime job identity, used for deterministic paths and checkpoint metadata.
    job: JobConfig = attrs.field(factory=JobConfig)

    # Runtime configs.
    runtime: RuntimeConfig = attrs.field(factory=RuntimeConfig)

    if USE_MEGATRON:
        # Megatron-Core configs
        model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)
    else:
        model_parallel: None = None

    # Checkpointer configs.
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    # enable upload reproducible setup to s3
    upload_reproducible_setup: bool = False

    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def validate(self) -> None:
        """Validate that the config has all required fields."""

        # broadcast job.name across all ranks to make sure it is consistent
        # otherwise, unaligned job names leads unaligned path to save checkpoints
        job_name_tensor = torch.ByteTensor(bytearray(self.job.name, "utf-8")).cuda()
        distributed.broadcast(job_name_tensor, 0)
        self.job.name = job_name_tensor.cpu().numpy().tobytes().decode("utf-8")

        assert self.job.project != ""
        assert self.job.group != ""
        assert self.job.name != ""
