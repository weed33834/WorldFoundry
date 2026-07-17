"""DINO v1 hub entrypoints."""

from .backbones import (
    dino_resnet50,
    dino_vitb8,
    dino_vitb16,
    dino_vits8,
    dino_vits16,
)

__all__ = [
    "dino_resnet50",
    "dino_vitb8",
    "dino_vitb16",
    "dino_vits8",
    "dino_vits16",
]
