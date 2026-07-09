# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> image -> sana -> diffusion -> model -> dc_ae -> efficientvit -> models -> utils -> list.py functionality."""

from typing import Any, Optional, Union

__all__ = [
    "list_sum",
    "list_mean",
    "weighted_list_sum",
    "list_join",
    "val2list",
    "val2tuple",
    "squeeze_list",
]


def list_sum(x: list) -> Any:
    """List sum.

    Args:
        x: The x.

    Returns:
        The return value.
    """
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> Any:
    """List mean.

    Args:
        x: The x.

    Returns:
        The return value.
    """
    return list_sum(x) / len(x)


def weighted_list_sum(x: list, weights: list) -> Any:
    """Weighted list sum.

    Args:
        x: The x.
        weights: The weights.

    Returns:
        The return value.
    """
    assert len(x) == len(weights)
    return x[0] * weights[0] if len(x) == 1 else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])


def list_join(x: list, sep="\t", format_str="%s") -> str:
    """List join.

    Args:
        x: The x.
        sep: The sep.
        format_str: The format str.

    Returns:
        The return value.
    """
    return sep.join([format_str % val for val in x])


def val2list(x: Union[list, tuple, Any], repeat_time=1) -> list:
    """Val2list.

    Args:
        x: The x.
        repeat_time: The repeat time.

    Returns:
        The return value.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[list, tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """Val2tuple.

    Args:
        x: The x.
        min_len: The min len.
        idx_repeat: The idx repeat.

    Returns:
        The return value.
    """
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def squeeze_list(x: Optional[list]) -> Union[list, Any]:
    """Squeeze list.

    Args:
        x: The x.

    Returns:
        The return value.
    """
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x
