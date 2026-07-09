"""Cosmos 2.5 checkpoint discovery (shared implementation)."""

from worldfoundry.base_models.diffusion_model.video.cosmos.shared.checkpoint_artifacts import (
    candidate_repo_dirs,
    find_existing_child,
    find_local_artifact_path,
    resolve_local_artifact_path as _resolve_local_artifact_path,
)


def resolve_local_artifact_path(repo_id: str, relative_paths=()):
    return _resolve_local_artifact_path(repo_id, relative_paths, family_label="Cosmos")


__all__ = [
    "candidate_repo_dirs",
    "find_existing_child",
    "find_local_artifact_path",
    "resolve_local_artifact_path",
]
