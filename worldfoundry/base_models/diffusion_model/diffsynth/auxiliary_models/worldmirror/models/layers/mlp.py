# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


"""Module for base_models -> diffusion_model -> diffsynth -> auxiliary_models -> worldmirror -> models -> layers -> mlp.py functionality."""

from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    """Mlp implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            drop: The drop.
            bias: The bias.

        Returns:
            The return value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
