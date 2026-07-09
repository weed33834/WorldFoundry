"""Runtime package boundary for ReCamMaster assets."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    return Path(__file__).resolve().parent


__all__ = ["runtime_root"]
