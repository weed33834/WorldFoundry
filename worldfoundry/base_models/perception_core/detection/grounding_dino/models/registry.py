"""Module for base_models -> perception_core -> detection -> grounding_dino -> models -> registry.py functionality."""

from ..registry_core import Registry

MODULE_BUILD_FUNCS = Registry("model build functions")

__all__ = ["MODULE_BUILD_FUNCS", "Registry"]
