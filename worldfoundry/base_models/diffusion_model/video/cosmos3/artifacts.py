"""Cosmos 3 checkpoint discovery (shared implementation)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.base_models.diffusion_model.video.cosmos.shared.checkpoint_artifacts import (
    candidate_repo_dirs,
    checkpoint_roots,
    find_existing_child,
    find_local_artifact_path,
    resolve_local_artifact_path as _resolve_local_artifact_path,
)

DEFAULT_COSMOS3_REPO_ID = "nvidia/Cosmos3-Nano"
DEFAULT_COSMOS3_SUPER_REPO_ID = "nvidia/Cosmos3-Super"
DEFAULT_COSMOS3_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
DEFAULT_COSMOS3_SUPER_REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"

_REPO_REVISIONS = {
    DEFAULT_COSMOS3_REPO_ID: DEFAULT_COSMOS3_REVISION,
    DEFAULT_COSMOS3_SUPER_REPO_ID: DEFAULT_COSMOS3_SUPER_REVISION,
}

COSMOS3_MODEL_SOURCE_KEYS = frozenset(
    {"pretrained_model_path", "checkpoint_path", "model_path", "repo_id", "repo_root"}
)
_MODEL_SOURCE_KEY_PRECEDENCE = (
    "pretrained_model_path",
    "checkpoint_path",
    "model_path",
    "repo_id",
    "repo_root",
)
COSMOS3_LOADER_METADATA_KEYS = frozenset(
    {
        "adapter",
        "adapter_target",
        "acquisition_root",
        "device",
        "hf_models_root",
        "manifest_path",
        "model_adapter",
        "model_id",
        "pipeline_binding",
        "pipeline_target",
        "profile_id",
        "profile_path",
        "revision",
        "runtime_profile",
        "variant_id",
    }
)

_VARIANT_REPO_IDS = {
    "cosmos3": DEFAULT_COSMOS3_REPO_ID,
    "cosmos-3": DEFAULT_COSMOS3_REPO_ID,
    "cosmos3-nano": DEFAULT_COSMOS3_REPO_ID,
    "cosmos-3-nano": DEFAULT_COSMOS3_REPO_ID,
    "nano": DEFAULT_COSMOS3_REPO_ID,
    DEFAULT_COSMOS3_REPO_ID.lower(): DEFAULT_COSMOS3_REPO_ID,
    "cosmos3-super": DEFAULT_COSMOS3_SUPER_REPO_ID,
    "cosmos-3-super": DEFAULT_COSMOS3_SUPER_REPO_ID,
    "super": DEFAULT_COSMOS3_SUPER_REPO_ID,
    DEFAULT_COSMOS3_SUPER_REPO_ID.lower(): DEFAULT_COSMOS3_SUPER_REPO_ID,
}


def _selector_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("id") or value.get("profile_id") or value.get("model_id")
    return str(value or "").strip()


def cosmos3_repo_id_for_selector(selector: Any) -> str | None:
    """Map a Cosmos3 model/profile selector to its official checkpoint repo."""

    text = _selector_text(selector)
    return _VARIANT_REPO_IDS.get(text.lower().replace("_", "-")) if text else None


def cosmos3_revision_for_repo_id(repo_id: str) -> str | None:
    """Return the immutable revision pinned for an official Cosmos3 repository."""

    return _REPO_REVISIONS.get(repo_id)


def checkpoint_revision(path: str | Path) -> str | None:
    """Read a verifiable revision from an HFD directory or native Hub snapshot."""

    candidate = Path(path).expanduser()
    metadata_path = candidate / ".hfd" / "repo_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        revision = str(metadata.get("sha") or "").strip()
        return revision or None

    # Native huggingface_hub cache paths end in snapshots/<immutable SHA>.
    if candidate.parent.name == "snapshots" and len(candidate.name) == 40:
        return candidate.name
    return None


def _checkpoint_repo_id(path: str | Path) -> str | None:
    metadata_path = Path(path).expanduser() / ".hfd" / "repo_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    repo_id = str(metadata.get("id") or metadata.get("repo_id") or "").strip()
    return repo_id or None


def candidate_repo_dirs_at_revision(repo_id: str, revision: str) -> list[Path]:
    """Return only locally cached candidates that prove they match ``revision``."""

    candidates = candidate_repo_dirs(repo_id)
    repo_name = repo_id.rsplit("/", 1)[-1]
    revision_suffix = revision[:8]
    search_roots = {*checkpoint_roots(), *(candidate.parent for candidate in candidates)}
    for parent in search_roots:
        for name in (f"{repo_name}-{revision_suffix}", f"{repo_id.replace('/', '--')}-{revision_suffix}"):
            candidate = parent / name
            if candidate.exists() and candidate.resolve() not in candidates:
                candidates.append(candidate.resolve())
    return [candidate for candidate in candidates if checkpoint_revision(candidate) == revision]


def resolve_cosmos3_variant_id(
    request: Mapping[str, Any] | None = None,
    *,
    model_id: str | None = None,
    model_source: str | None = None,
) -> str:
    """Resolve the canonical Nano/Super profile from WorldFoundry loader metadata."""

    request = request or {}
    source_repo = cosmos3_repo_id_for_selector(model_source)
    if source_repo is None and model_source:
        source_repo = cosmos3_repo_id_for_selector(_checkpoint_repo_id(model_source))
    if source_repo == DEFAULT_COSMOS3_SUPER_REPO_ID:
        return "cosmos3-super"
    if source_repo == DEFAULT_COSMOS3_REPO_ID:
        return "cosmos3-nano"

    source_name = Path(str(model_source or "")).name.lower().replace("_", "-")
    if "cosmos3-super" in source_name or "cosmos-3-super" in source_name:
        return "cosmos3-super"
    if "cosmos3-nano" in source_name or "cosmos-3-nano" in source_name:
        return "cosmos3-nano"

    selectors = (
        request.get("variant_id"),
        request.get("profile_id"),
        request.get("runtime_profile"),
        model_id,
        request.get("model_id"),
    )
    for selector in selectors:
        repo_id = cosmos3_repo_id_for_selector(selector)
        if repo_id == DEFAULT_COSMOS3_SUPER_REPO_ID:
            return "cosmos3-super"
        if repo_id == DEFAULT_COSMOS3_REPO_ID:
            return "cosmos3-nano"
        text = _selector_text(selector).lower().replace("_", "-")
        if text.startswith(("cosmos3-", "cosmos-3-")):
            raise ValueError(f"Unsupported Cosmos3 variant/profile: {selector!r}")
    return "cosmos3-nano"


def resolve_cosmos3_model_source(
    model_path: str | Mapping[str, Any] | None = None,
    *,
    model_id: str | None = None,
) -> str:
    """Resolve an explicit path/repo or route a WorldFoundry Nano/Super request."""

    if isinstance(model_path, Mapping):
        for key in _MODEL_SOURCE_KEY_PRECEDENCE:
            value = model_path.get(key)
            if value is not None and not isinstance(value, Mapping) and str(value).strip():
                text = str(value).strip()
                mapped = cosmos3_repo_id_for_selector(text)
                normalized = text.lower().replace("_", "-")
                if mapped is None and normalized.startswith(("cosmos3-", "cosmos-3-")):
                    raise ValueError(f"Unsupported Cosmos3 checkpoint selector: {text!r}")
                return mapped or text
        variant_id = resolve_cosmos3_variant_id(model_path, model_id=model_id)
        return DEFAULT_COSMOS3_SUPER_REPO_ID if variant_id == "cosmos3-super" else DEFAULT_COSMOS3_REPO_ID

    if model_path is not None and str(model_path).strip():
        text = str(model_path).strip()
        mapped = cosmos3_repo_id_for_selector(text)
        normalized = text.lower().replace("_", "-")
        if mapped is None and normalized.startswith(("cosmos3-", "cosmos-3-")):
            raise ValueError(f"Unsupported Cosmos3 checkpoint selector: {text!r}")
        return mapped or text

    variant_id = resolve_cosmos3_variant_id(model_id=model_id)
    return DEFAULT_COSMOS3_SUPER_REPO_ID if variant_id == "cosmos3-super" else DEFAULT_COSMOS3_REPO_ID


def strip_cosmos3_loader_metadata(options: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only model-loader options after consuming WorldFoundry metadata."""

    consumed = COSMOS3_MODEL_SOURCE_KEYS | COSMOS3_LOADER_METADATA_KEYS
    return {key: value for key, value in options.items() if key not in consumed}


def resolve_local_artifact_path(repo_id: str, relative_paths=()):
    return _resolve_local_artifact_path(repo_id, relative_paths, family_label="Cosmos3")


__all__ = [
    "DEFAULT_COSMOS3_REPO_ID",
    "DEFAULT_COSMOS3_REVISION",
    "DEFAULT_COSMOS3_SUPER_REPO_ID",
    "DEFAULT_COSMOS3_SUPER_REVISION",
    "COSMOS3_LOADER_METADATA_KEYS",
    "COSMOS3_MODEL_SOURCE_KEYS",
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
