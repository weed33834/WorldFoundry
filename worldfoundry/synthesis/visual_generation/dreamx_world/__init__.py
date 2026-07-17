"""In-tree DreamX-World interactive generation runtimes."""

from __future__ import annotations

from typing import Any

__all__ = ["DreamXWorldARRealtimeSession", "DreamXWorldRealtimeSession"]


def __getattr__(name: str) -> Any:
    if name == "DreamXWorldARRealtimeSession":
        from .ar_realtime import DreamXWorldARRealtimeSession

        return DreamXWorldARRealtimeSession
    if name == "DreamXWorldRealtimeSession":
        from .realtime import DreamXWorldRealtimeSession

        return DreamXWorldRealtimeSession
    raise AttributeError(name)
