"""Inference-only, in-tree integration for Tencent Hy-Embodied-0.5-VLA."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "HyDualTower": (".modeling_dual_tower", "HyDualTower"),
    "HyDualTowerConfig": (".modeling_dual_tower", "HyDualTowerConfig"),
    "HyEmbodiedVLARuntime": (".runtime", "HyEmbodiedVLARuntime"),
    "HyEmbodiedVLARuntimeConfig": (".runtime", "HyEmbodiedVLARuntimeConfig"),
    "HyEmbodiedVLASynthesis": (".hy_embodied_vla_synthesis", "HyEmbodiedVLASynthesis"),
    "HyVLA": (".modeling_hy_vla", "HyVLA"),
    "HyVLAConfig": (".configuration_hy_vla", "HyVLAConfig"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load heavy torch/transformers modules only when their symbol is used."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
