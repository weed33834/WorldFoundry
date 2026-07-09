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

"""Module for base_models -> three_dimensions -> general_3d -> vipe -> ext -> specs.py functionality."""

import os
from pathlib import Path


def _csrc_path() -> Path:
    """Helper function to csrc path.

    Returns:
        The return value.
    """
    return Path(__file__).parent.parent.parent / "csrc"


def get_sources() -> list[str]:
    """Get sources.

    Returns:
        The return value.
    """
    csrc_path = _csrc_path()
    return [str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]]


def _eigen_include_flags() -> list[str]:
    """Helper function to eigen include flags.

    Returns:
        The return value.
    """
    if os.environ.get("USE_SYSTEM_EIGEN", "0") == "1":
        return []

    candidates = []
    if os.environ.get("EIGEN3_INCLUDE_DIR"):
        candidates.append(Path(os.environ["EIGEN3_INCLUDE_DIR"]))
    if os.environ.get("EIGEN_INCLUDE_DIR"):
        candidates.append(Path(os.environ["EIGEN_INCLUDE_DIR"]))
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        candidates.extend([conda_prefix / "include" / "eigen3", conda_prefix / "include"])
    candidates.extend([Path("/usr/include/eigen3"), Path("/usr/local/include/eigen3")])

    for include_path in candidates:
        if (include_path / "Eigen").exists():
            return ["-isystem", str(include_path.resolve())]

    raise RuntimeError(
        "Eigen headers are required to build the VIPE extension. Install Eigen "
        "(for example libeigen3-dev or conda-forge::eigen) or set EIGEN3_INCLUDE_DIR."
    )


def _additional_include_flags() -> list[str]:
    """Helper function to additional include flags.

    Returns:
        The return value.
    """
    flags = _eigen_include_flags()
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        include_paths = [
            conda_prefix / "include",
            conda_prefix / "nvvm" / "include",
            *sorted((conda_prefix / "targets").glob("*/include")),
        ]
        for include_path in include_paths:
            if include_path.exists():
                flags += ["-isystem", str(include_path)]
    return flags


def get_cpp_flags() -> list[str]:
    """Get cpp flags.

    Returns:
        The return value.
    """
    return ["-O3", "-DWITH_CUDA"] + _additional_include_flags()


def get_cuda_flags() -> list[str]:
    """Get cuda flags.

    Returns:
        The return value.
    """
    return ["-O3", "-DWITH_CUDA", "--use_fast_math"] + _additional_include_flags()
