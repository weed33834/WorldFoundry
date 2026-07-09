"""Module for base_models -> perception_core -> segment -> efficient_sam -> mlp.py functionality."""

from typing import Type

from torch import nn


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLPBlock(nn.Module):
    """Mlp block implementation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: Type[nn.Module],
    ) -> None:
        """Init.

        Args:
            input_dim: The input dim.
            hidden_dim: The hidden dim.
            output_dim: The output dim.
            num_layers: The num layers.
            act: The act.

        Returns:
            The return value.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(n, k), act())
            for n, k in zip([input_dim] + h, [hidden_dim] * num_layers)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)
