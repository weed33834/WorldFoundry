"""In-tree SkyReels-V2 official inference runtime."""

from __future__ import annotations

from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
ENTRYPOINT = RUNTIME_DIR / "inference.py"

__all__ = ["ENTRYPOINT", "RUNTIME_DIR"]
