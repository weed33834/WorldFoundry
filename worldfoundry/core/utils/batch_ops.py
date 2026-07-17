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

# Functions for performing operations with broadcasting to the right axis
#
# Example
# input1: tensor of size (N1, N2)
# input2: tensor of size (N1, N2, N3, N4)
# batch_mul(input1, input2) = input1[:, :, None, None] * input2
#
# If the common dimensions don't match, we raise an assertion error.

from collections.abc import Sequence

import torch
from torch import Tensor


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_add(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x + y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def batch_sub(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x - y


def batch_div(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x / y


def stack_or_pad_tensors(
    values: Sequence[Tensor],
    *,
    padding_value: float | int | bool | None = None,
) -> Tensor:
    """Stack equal tensors or right-pad variable-length 1-D tensors.

    This is the common collation shape used by multimodal policies whose dense
    image/state inputs are fixed while token and mask lengths may vary.  More
    complex shape mismatches are rejected instead of being padded silently.
    """

    if not values:
        raise ValueError("values must contain at least one tensor")
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("all values must be torch tensors")
    shapes = [tuple(value.shape) for value in values]
    if len(set(shapes)) == 1:
        return torch.stack(tuple(values), dim=0)
    if not all(value.ndim == 1 for value in values):
        raise ValueError(f"cannot stack tensors with incompatible shapes: {shapes}")
    if padding_value is None:
        padding_value = False if values[0].dtype == torch.bool else 0
    return torch.nn.utils.rnn.pad_sequence(
        tuple(values),
        batch_first=True,
        padding_value=padding_value,
    )


__all__ = [
    "batch_add",
    "batch_div",
    "batch_mul",
    "batch_sub",
    "common_broadcast",
    "stack_or_pad_tensors",
]
