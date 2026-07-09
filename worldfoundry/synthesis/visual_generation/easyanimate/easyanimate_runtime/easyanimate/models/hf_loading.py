from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def looks_like_hf_repo_id(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return False
    return value.count("/") == 1


def resolve_pretrained_subfolder(pretrained_model_path: str, subfolder: str | None = None) -> str:
    """Return a local directory for EasyAnimate's file-based loaders.

    Local paths are used as-is. Hugging Face repo ids are resolved through
    huggingface_hub without overriding cache_dir, so HF_HOME/HF_HUB_CACHE and
    the native ~/.cache/huggingface default behavior remain in charge.
    """

    model_path = str(pretrained_model_path)
    expanded = str(Path(model_path).expanduser())
    local_root = expanded if Path(expanded).exists() else model_path
    local_subfolder = os.path.join(local_root, subfolder) if subfolder else local_root
    if os.path.isdir(local_subfolder):
        return local_subfolder
    if not looks_like_hf_repo_id(model_path):
        return local_subfolder

    allow_patterns = None
    if subfolder:
        allow_patterns = [f"{subfolder}/*"]
    snapshot_path = snapshot_download(model_path, allow_patterns=allow_patterns)
    return os.path.join(snapshot_path, subfolder) if subfolder else snapshot_path
