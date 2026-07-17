"""Module for base_models -> diffusion_model -> video -> cosmos3 -> __init__.py functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_COSMOS3_REPO_ID,
    DEFAULT_COSMOS3_REVISION,
    DEFAULT_COSMOS3_SUPER_REPO_ID,
    DEFAULT_COSMOS3_SUPER_REVISION,
    COSMOS3_LOADER_METADATA_KEYS,
    COSMOS3_MODEL_SOURCE_KEYS,
    candidate_repo_dirs,
    candidate_repo_dirs_at_revision,
    checkpoint_revision,
    cosmos3_repo_id_for_selector,
    cosmos3_revision_for_repo_id,
    find_existing_child,
    find_local_artifact_path,
    resolve_local_artifact_path,
    resolve_cosmos3_model_source,
    resolve_cosmos3_variant_id,
    strip_cosmos3_loader_metadata,
)


SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parent


def __getattr__(name: str) -> Any:
    """Getattr.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    if name in {"Cosmos3Runtime", "Cosmos3RuntimeOutput", "Cosmos3RuntimePlan"}:
        from .worldfoundry_runtime import Cosmos3Runtime, Cosmos3RuntimeOutput, Cosmos3RuntimePlan

        return {
            "Cosmos3Runtime": Cosmos3Runtime,
            "Cosmos3RuntimeOutput": Cosmos3RuntimeOutput,
            "Cosmos3RuntimePlan": Cosmos3RuntimePlan,
        }[name]
    if name in {"CosmosActionCondition", "Cosmos3AVAEAudioTokenizer"}:
        from .diffusers_cosmos3 import Cosmos3AVAEAudioTokenizer, CosmosActionCondition

        return {
            "CosmosActionCondition": CosmosActionCondition,
            "Cosmos3AVAEAudioTokenizer": Cosmos3AVAEAudioTokenizer,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DEFAULT_COSMOS3_REPO_ID",
    "DEFAULT_COSMOS3_REVISION",
    "DEFAULT_COSMOS3_SUPER_REPO_ID",
    "DEFAULT_COSMOS3_SUPER_REVISION",
    "COSMOS3_LOADER_METADATA_KEYS",
    "COSMOS3_MODEL_SOURCE_KEYS",
    "SOURCE_PACKAGE_ROOT",
    "Cosmos3Runtime",
    "Cosmos3RuntimeOutput",
    "Cosmos3RuntimePlan",
    "Cosmos3AVAEAudioTokenizer",
    "CosmosActionCondition",
    "candidate_repo_dirs",
    "candidate_repo_dirs_at_revision",
    "checkpoint_revision",
    "cosmos3_repo_id_for_selector",
    "cosmos3_revision_for_repo_id",
    "find_existing_child",
    "find_local_artifact_path",
    "resolve_local_artifact_path",
    "resolve_cosmos3_model_source",
    "resolve_cosmos3_variant_id",
    "strip_cosmos3_loader_metadata",
]
