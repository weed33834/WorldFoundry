"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> global_cfg.py functionality."""

from typing import Optional

from omegaconf import DictConfig

cfg: Optional[DictConfig] = None


def get_cfg() -> DictConfig:
    """Get cfg.

    Returns:
        The return value.
    """
    global cfg
    return cfg


def set_cfg(new_cfg: DictConfig) -> None:
    """Set cfg.

    Args:
        new_cfg: The new cfg.

    Returns:
        The return value.
    """
    global cfg
    cfg = new_cfg


def get_seed() -> int:
    """Get seed.

    Returns:
        The return value.
    """
    return cfg.seed
