"""Inference-only CogACT model implementation.

The public objects are resolved lazily so importing an individual modeling
component (for example the DiT action head) never initializes the 7B VLM or
loads checkpoint assets as an import-time side effect.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CogACT", "load_vla"]


def __getattr__(name: str) -> Any:
    if name == "CogACT":
        from .policy import CogACT

        return CogACT
    if name == "load_vla":
        from .loader import load_vla

        return load_vla
    raise AttributeError(name)
