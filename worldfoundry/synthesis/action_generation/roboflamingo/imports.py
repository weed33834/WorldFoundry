# Inference-only RoboFlamingo source retained in-tree.
from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo import (
    PACKAGE_ROOT as OPEN_FLAMINGO_ROOT,
    ensure_import_paths as ensure_open_flamingo_import_paths,
)


RUNTIME_ROOT = Path(__file__).resolve().parent


def ensure_runtime_import_paths() -> tuple[Path, ...]:
    """Expose RoboFlamingo runtime code and shared OpenFlamingo base code."""

    ensure_open_flamingo_import_paths()
    return (OPEN_FLAMINGO_ROOT,)


__all__ = ["RUNTIME_ROOT", "ensure_runtime_import_paths"]
