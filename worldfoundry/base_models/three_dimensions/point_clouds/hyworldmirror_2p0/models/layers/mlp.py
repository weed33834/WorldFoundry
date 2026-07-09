"""Module for base_models -> three_dimensions -> point_clouds -> hyworldmirror_2p0 -> models -> layers -> mlp.py functionality."""

import torch
from torch import Tensor

from worldfoundry.core.nn import Mlp


class MlpFP32(Mlp):
    """Mlp implementation."""
    @staticmethod
    def map_to_args_to_float(args, kwargs):
        """Map to args to float.

        Args:
            args: The args.
            kwargs: The kwargs.
        """
        args = tuple(
            torch.float32 if isinstance(arg, torch.dtype) else arg
            for arg in args
        )
        kwargs = dict(kwargs)
        for key in kwargs:
            if key == "dtype":
                kwargs[key] = torch.float32
        return args, kwargs

    def to(self, *args, **kwargs):
        """To."""
        self.fc1 = self.fc1.to(*args, **kwargs)
        args, kwargs = self.map_to_args_to_float(args, kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        return self
    
    def forward_infer(self, x):
        """Forward infer.

        Args:
            x: The x.
        """
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x * 2**-0.5))
        x = self.fc2(x.float())
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        return self.forward_infer(x)
