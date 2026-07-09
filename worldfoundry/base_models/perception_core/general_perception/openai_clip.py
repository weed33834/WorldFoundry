"""Vendored OpenAI CLIP runtime helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(__file__).resolve().parent / "openai_clip_runtime"


def add_runtime_to_path() -> Path:
    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_ROOT))
    return RUNTIME_ROOT


def load(*args: Any, **kwargs: Any):
    add_runtime_to_path()
    import clip

    return clip.load(*args, **kwargs)


__all__ = ["RUNTIME_ROOT", "add_runtime_to_path", "load"]
