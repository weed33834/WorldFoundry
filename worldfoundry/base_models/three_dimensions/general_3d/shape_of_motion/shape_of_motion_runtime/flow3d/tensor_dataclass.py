"""Module for base_models -> three_dimensions -> general_3d -> shape_of_motion -> shape_of_motion_runtime -> flow3d -> tensor_dataclass.py functionality."""

from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
from typing_extensions import Self

TensorDataclassT = TypeVar("T", bound="TensorDataclass")


class TensorDataclass:
    """A lighter version of nerfstudio's TensorDataclass:
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/tensor_dataclass.py
    """

    def __getitem__(self, key) -> Self:
        """Getitem.

        Args:
            key: The key.

        Returns:
            The return value.
        """
        return self.map(lambda x: x[key])

    def to(self, device: torch.device | str) -> Self:
        """Move the tensors in the dataclass to the given device.

        Args:
            device: The device to move to.

        Returns:
            A new dataclass.
        """
        return self.map(lambda x: x.to(device))

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """Apply a function to all tensors in the dataclass.

        Also recurses into lists, tuples, and dictionaries.

        Args:
            fn: The function to apply to each tensor.

        Returns:
            A new dataclass.
        """

        MapT = TypeVar("MapT")

        def _map_impl(
            fn: Callable[[torch.Tensor], torch.Tensor],
            val: MapT,
        ) -> MapT:
            """Helper function to map impl.

            Args:
                fn: The fn.
                val: The val.

            Returns:
                The return value.
            """
            if isinstance(val, torch.Tensor):
                return fn(val)
            elif isinstance(val, TensorDataclass):
                return type(val)(**_map_impl(fn, vars(val)))
            elif isinstance(val, (list, tuple)):
                return type(val)(_map_impl(fn, v) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support.
                return {k: _map_impl(fn, v) for k, v in val.items()}  # type: ignore
            else:
                return val

        return _map_impl(fn, self)


@dataclass
class TrackObservations(TensorDataclass):
    """Track observations implementation."""
    xyz: torch.Tensor
    visibles: torch.Tensor
    invisibles: torch.Tensor
    confidences: torch.Tensor
    colors: torch.Tensor

    def check_sizes(self) -> bool:
        """Check sizes.

        Returns:
            The return value.
        """
        dims = self.xyz.shape[:-1]
        return (
            self.visibles.shape == dims
            and self.invisibles.shape == dims
            and self.confidences.shape == dims
            and self.colors.shape[:-1] == dims[:-1]
            and self.xyz.shape[-1] == 3
            and self.colors.shape[-1] == 3
        )

    def filter_valid(self, valid_mask: torch.Tensor) -> Self:
        """Filter valid.

        Args:
            valid_mask: The valid mask.

        Returns:
            The return value.
        """
        return self.map(lambda x: x[valid_mask])


@dataclass
class StaticObservations(TensorDataclass):
    """Static observations implementation."""
    xyz: torch.Tensor
    normals: torch.Tensor
    colors: torch.Tensor

    def check_sizes(self) -> bool:
        """Check sizes.

        Returns:
            The return value.
        """
        dims = self.xyz.shape
        return self.normals.shape == dims and self.colors.shape == dims

    def filter_valid(self, valid_mask: torch.Tensor) -> Self:
        """Filter valid.

        Args:
            valid_mask: The valid mask.

        Returns:
            The return value.
        """
        return self.map(lambda x: x[valid_mask])
