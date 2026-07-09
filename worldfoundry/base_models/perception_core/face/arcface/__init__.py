"""Minimal ArcFace inference components shared by benchmark runners."""

from .resnet import ResNetFace, resnet_face18

__all__ = ["ResNetFace", "resnet_face18"]
