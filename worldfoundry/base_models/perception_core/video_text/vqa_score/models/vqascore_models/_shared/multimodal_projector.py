import re

import torch.nn as nn


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, int(mlp_gelu_match.group(1))):
            modules.extend((nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"Unknown projector type: {projector_type}")


__all__ = ["IdentityMap", "SimpleResBlock", "build_vision_projector"]
