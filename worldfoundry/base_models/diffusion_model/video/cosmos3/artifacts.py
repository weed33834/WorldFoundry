"""Cosmos 3 checkpoint discovery (shared implementation)."""

from __future__ import annotations

from worldfoundry.base_models.diffusion_model.video.cosmos.shared.checkpoint_artifacts import (
    candidate_repo_dirs,
    find_existing_child,
    find_local_artifact_path,
    resolve_local_artifact_path as _resolve_local_artifact_path,
)

DEFAULT_COSMOS3_REPO_ID = "nvidia/Cosmos3-Nano"
DEFAULT_COSMOS3_SUPER_REPO_ID = "nvidia/Cosmos3-Super"


def resolve_local_artifact_path(repo_id: str, relative_paths=()):
    return _resolve_local_artifact_path(repo_id, relative_paths, family_label="Cosmos3")


__all__ = [
    "DEFAULT_COSMOS3_REPO_ID",
    "DEFAULT_COSMOS3_SUPER_REPO_ID",
    "candidate_repo_dirs",
    "find_existing_child",
    "find_local_artifact_path",
    "resolve_local_artifact_path",
]
