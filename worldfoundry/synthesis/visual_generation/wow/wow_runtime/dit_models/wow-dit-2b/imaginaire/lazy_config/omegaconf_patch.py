"""OmegaConf compatibility shim for the in-tree lazy config loader."""

from __future__ import annotations

from omegaconf import OmegaConf, SCMode


def to_object(cfg):
    """Convert OmegaConf containers to Python objects."""

    return OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
        structured_config_mode=SCMode.INSTANTIATE,
    )
