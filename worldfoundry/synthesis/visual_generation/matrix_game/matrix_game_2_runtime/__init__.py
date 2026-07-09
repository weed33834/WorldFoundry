"""Matrix-Game-2 in-tree inference runtime."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    return Path(__file__).resolve().parent


def __getattr__(name: str):
    if name == "MatrixGame2Runtime":
        from .worldfoundry_runtime import MatrixGame2Runtime

        return MatrixGame2Runtime
    raise AttributeError(name)


__all__ = ["MatrixGame2Runtime", "runtime_root"]
