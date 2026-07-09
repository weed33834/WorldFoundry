"""Module for base_models -> diffusion_model -> video -> cosmos3 -> __init__.py functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_COSMOS3_REPO_ID,
    DEFAULT_COSMOS3_SUPER_REPO_ID,
    candidate_repo_dirs,
    find_existing_child,
    find_local_artifact_path,
    resolve_local_artifact_path,
)


SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parent


def __getattr__(name: str) -> Any:
    """Getattr.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    if name in {"Cosmos3Runtime", "Cosmos3RuntimePlan"}:
        from .worldfoundry_runtime import Cosmos3Runtime, Cosmos3RuntimePlan

        return {"Cosmos3Runtime": Cosmos3Runtime, "Cosmos3RuntimePlan": Cosmos3RuntimePlan}[name]
    raise AttributeError(name)


__all__ = [
    "DEFAULT_COSMOS3_REPO_ID",
    "DEFAULT_COSMOS3_SUPER_REPO_ID",
    "SOURCE_PACKAGE_ROOT",
    "Cosmos3Runtime",
    "Cosmos3RuntimePlan",
    "candidate_repo_dirs",
    "find_existing_child",
    "find_local_artifact_path",
    "resolve_local_artifact_path",
]
