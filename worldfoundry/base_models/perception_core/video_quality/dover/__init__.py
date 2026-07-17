"""DOVER video quality runtime."""

from .paths import checkpoint_path, config_path, package_root
from .runtime import DOVERTechnicalScorer

__all__ = ["DOVERTechnicalScorer", "checkpoint_path", "config_path", "package_root"]
