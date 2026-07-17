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

"""Module for base_models -> diffusion_model -> image -> sana -> diffusion -> model -> dc_ae -> efficientvit -> models -> nn -> norm.py functionality."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ...models.nn.triton_rms_norm import TritonRMSNorm2dFunc
from ...models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    """Layer norm d implementation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    """Triton rms norm d implementation."""
    def zero_out(self):
        """Zero out."""
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        input_numel = x.numel()
        if input_numel >= 1 << 31:
            num_chunks = (input_numel - 1) // (1 << 31) + 1
            output = []
            for x_chunk in x.chunk(num_chunks, dim=2):
                output.append(TritonRMSNorm2dFunc.apply(x_chunk.contiguous(), self.weight, self.bias, self.eps))
            output = torch.cat(output, dim=2)
            return output
        else:
            return TritonRMSNorm2dFunc.apply(x.contiguous(), self.weight, self.bias, self.eps)


class RMSNorm2d(nn.Module):
    """Rms norm d implementation."""
    def __init__(
        self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ) -> None:
        """Init.

        Args:
            num_features: The num features.
            eps: The eps.
            elementwise_affine: The elementwise affine.
            bias: The bias.

        Returns:
            The return value.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class RMSNorm3d(RMSNorm2d):
    """Rms norm d implementation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
    "rms2d": RMSNorm2d,
    "rms3d": RMSNorm3d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    """Build norm.

    Args:
        name: The name.
        num_features: The num features.

    Returns:
        The return value.
    """
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    """Set norm eps.

    Args:
        model: The model.
        eps: The eps.

    Returns:
        The return value.
    """
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
