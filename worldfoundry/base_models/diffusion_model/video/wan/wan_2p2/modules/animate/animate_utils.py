# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p2 -> modules -> animate -> animate_utils.py functionality."""

import torch
import numbers
from peft import LoraConfig


def get_loraconfig(transformer, rank=128, alpha=128, init_lora_weights="gaussian"):
    """Get loraconfig.

    Args:
        transformer: The transformer.
        rank: The rank.
        alpha: The alpha.
        init_lora_weights: The init lora weights.
    """
    target_modules = []
    for name, module in transformer.named_modules():
        if "blocks" in name and "face" not in name and "modulation" not in name and isinstance(module, torch.nn.Linear):
            target_modules.append(name)

    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    return transformer_lora_config



class TensorList(object):
    """Tensor list implementation."""

    def __init__(self, tensors):
        """
        tensors: a list of torch.Tensor objects. No need to have uniform shape.
        """
        assert isinstance(tensors, (list, tuple))
        assert all(isinstance(u, torch.Tensor) for u in tensors)
        assert len(set([u.ndim for u in tensors])) == 1
        assert len(set([u.dtype for u in tensors])) == 1
        assert len(set([u.device for u in tensors])) == 1
        self.tensors = tensors
    
    def to(self, *args, **kwargs):
        """To."""
        return TensorList([u.to(*args, **kwargs) for u in self.tensors])
    
    def size(self, dim):
        """Size.

        Args:
            dim: The dim.
        """
        assert dim == 0, 'only support get the 0th size'
        return len(self.tensors)
    
    def pow(self, *args, **kwargs):
        """Pow."""
        return TensorList([u.pow(*args, **kwargs) for u in self.tensors])
    
    def squeeze(self, dim):
        """Squeeze.

        Args:
            dim: The dim.
        """
        assert dim != 0
        if dim > 0:
            dim -= 1
        return TensorList([u.squeeze(dim) for u in self.tensors])
    
    def type(self, *args, **kwargs):
        """Type."""
        return TensorList([u.type(*args, **kwargs) for u in self.tensors])
    
    def type_as(self, other):
        """Type as.

        Args:
            other: The other.
        """
        assert isinstance(other, (torch.Tensor, TensorList))
        if isinstance(other, torch.Tensor):
            return TensorList([u.type_as(other) for u in self.tensors])
        else:
            return TensorList([u.type(other.dtype) for u in self.tensors])
    
    @property
    def dtype(self):
        """Dtype."""
        return self.tensors[0].dtype
    
    @property
    def device(self):
        """Device."""
        return self.tensors[0].device
    
    @property
    def ndim(self):
        """Ndim."""
        return 1 + self.tensors[0].ndim
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        return self.tensors[index]
    
    def __len__(self):
        """Len."""
        return len(self.tensors)
    
    def __add__(self, other):
        """Add.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u + v)
    
    def __radd__(self, other):
        """Radd.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v + u)
    
    def __sub__(self, other):
        """Sub.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u - v)
    
    def __rsub__(self, other):
        """Rsub.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v - u)
    
    def __mul__(self, other):
        """Mul.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u * v)
    
    def __rmul__(self, other):
        """Rmul.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v * u)
    
    def __floordiv__(self, other):
        """Floordiv.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u // v)
    
    def __truediv__(self, other):
        """Truediv.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u / v)
    
    def __rfloordiv__(self, other):
        """Rfloordiv.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v // u)
    
    def __rtruediv__(self, other):
        """Rtruediv.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v / u)
    
    def __pow__(self, other):
        """Pow.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: u ** v)
    
    def __rpow__(self, other):
        """Rpow.

        Args:
            other: The other.
        """
        return self._apply(other, lambda u, v: v ** u)
    
    def __neg__(self):
        """Neg."""
        return TensorList([-u for u in self.tensors])
    
    def __iter__(self):
        """Iter."""
        for tensor in self.tensors:
            yield tensor
    
    def __repr__(self):
        """Repr."""
        return 'TensorList: \n' + repr(self.tensors)

    def _apply(self, other, op):
        """Helper function to apply.

        Args:
            other: The other.
            op: The op.
        """
        if isinstance(other, (list, tuple, TensorList)) or (
            isinstance(other, torch.Tensor) and (
                other.numel() > 1 or other.ndim > 1
            )
        ):
            assert len(other) == len(self.tensors)
            return TensorList([op(u, v) for u, v in zip(self.tensors, other)])
        elif isinstance(other, numbers.Number) or (
            isinstance(other, torch.Tensor) and (
                other.numel() == 1 and other.ndim <= 1
            )
        ):
            return TensorList([op(u, other) for u in self.tensors])
        else:
            raise TypeError(
                f'unsupported operand for *: "TensorList" and "{type(other)}"'
            )
