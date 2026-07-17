"""OpenVLA-OFT public API with lazy profile loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["OpenVLAOFTSynthesis"]


def __getattr__(name: str) -> Any:
    if name != "OpenVLAOFTSynthesis":
        raise AttributeError(name)
    value = import_module(f"{__name__}.openvla_oft_synthesis").OpenVLAOFTSynthesis
    globals()[name] = value
    return value
