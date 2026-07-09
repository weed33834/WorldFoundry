"""Shared local checkpoint discovery for Cosmos-family diffusion models."""

from __future__ import annotations

import glob as glob_module
from pathlib import Path
from typing import Iterable

from worldfoundry.runtime import resolve_hfd_root

_CHECKPOINT_ROOTS = (resolve_hfd_root(),)


def _repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def candidate_repo_dirs(repo_id: str) -> list[Path]:
    """Return existing local candidates for a Hugging Face repo id or path."""

    path = Path(repo_id).expanduser()
    candidates = [path] if path.exists() else []
    candidates.extend(root / _repo_dir_name(repo_id) for root in _CHECKPOINT_ROOTS)
    candidates.extend(root / repo_id.split("/")[-1] for root in _CHECKPOINT_ROOTS)
    return [candidate.resolve() for candidate in candidates if candidate.exists()]


def find_existing_child(root: Path, relative_paths: Iterable[str]) -> Path | None:
    """Find the first existing relative child under ``root``."""

    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.exists():
            return candidate.resolve()
        if glob_module.has_magic(relative_path):
            matches = sorted(root.glob(relative_path))
            if matches:
                return matches[0].resolve()
    return None


def find_local_artifact_path(repo_id: str, relative_paths: Iterable[str] = ()) -> Path | None:
    """Locate a model artifact from local paths or the configured HFD cache."""

    for repo_dir in candidate_repo_dirs(repo_id):
        child = find_existing_child(repo_dir, relative_paths)
        if child is not None:
            return child
        if not relative_paths:
            return repo_dir
    return None


def resolve_local_artifact_path(
    repo_id: str,
    relative_paths: Iterable[str] = (),
    *,
    family_label: str = "Cosmos-family",
) -> Path:
    """Resolve a required local artifact or raise a cache miss error."""

    artifact_path = find_local_artifact_path(repo_id, relative_paths)
    if artifact_path is not None:
        return artifact_path

    searched = [
        str(path)
        for root in _CHECKPOINT_ROOTS
        for path in (root / _repo_dir_name(repo_id), root / repo_id.split("/")[-1])
    ]
    raise FileNotFoundError(
        f"{family_label} artifact is not available in the local WorldFoundry cache. "
        f"repo_id={repo_id!r}, searched={searched}, required_paths={list(relative_paths)!r}"
    )


__all__ = [
    "candidate_repo_dirs",
    "find_existing_child",
    "find_local_artifact_path",
    "resolve_local_artifact_path",
]
