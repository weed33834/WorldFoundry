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

"""Fixed feature switches for the inference-only runtime."""

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "y", "on"}


INTERNAL = _env_bool("COSMOS_INTERNAL")
VALIDATION = _env_bool("COSMOS_VALIDATION")
VERBOSE = _env_bool("COSMOS_VERBOSE")
EXPERIMENTAL_CHECKPOINTS = _env_bool("COSMOS_EXPERIMENTAL_CHECKPOINTS")
SMOKE = _env_bool("COSMOS_SMOKE")
TRAINING = False


@dataclass(frozen=True)
class Flags:
    internal: bool = INTERNAL
    validation: bool = VALIDATION
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()


__all__ = [
    "EXPERIMENTAL_CHECKPOINTS",
    "FLAGS",
    "INTERNAL",
    "SMOKE",
    "TRAINING",
    "VALIDATION",
    "VERBOSE",
    "Flags",
]
