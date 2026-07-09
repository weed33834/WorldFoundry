"""Shared utilities for native in-process embodied policy runtimes."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def option_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def option_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def option_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def resolve_source_workdir(
    options: Mapping[str, Any],
    repo_name: str,
    *,
    specific_env: str | None = None,
    default_subdir: str = "",
    in_tree_subdir: str | None = None,
    allow_external_override: bool | None = None,
) -> Path:
    source_subdir = str(options.get("source_subdir") or default_subdir or "")
    candidates: list[Path] = []

    repo_root = project_root().resolve()

    def _under_repo(path: Path) -> bool:
        try:
            path.expanduser().resolve().relative_to(repo_root)
            return True
        except ValueError:
            return False

    if in_tree_subdir:
        candidates.append(repo_root / in_tree_subdir)

    explicit = first_present(options, "source_repo", "official_source_repo", "repo_source")
    if explicit is not None:
        explicit_path = resolve_worldfoundry_path(str(explicit))
        if _under_repo(explicit_path):
            candidates.append(explicit_path)
        elif allow_external_override or option_bool(options.get("allow_external_source_repo"), False):
            candidates.append(explicit_path)

    if specific_env and os.environ.get(specific_env):
        env_path = resolve_worldfoundry_path(os.environ[specific_env])
        if _under_repo(env_path) or allow_external_override or option_bool(options.get("allow_external_source_repo"), False):
            candidates.append(env_path)

    seen: set[Path] = set()
    for root in candidates:
        workdir = (root / source_subdir).resolve() if source_subdir else root.resolve()
        if workdir in seen:
            continue
        seen.add(workdir)
        if workdir.exists():
            return workdir

    root = candidates[0] if candidates else repo_root / repo_name
    return (root / source_subdir).resolve() if source_subdir else root.resolve()


def ensure_import_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)
        importlib.invalidate_caches()
    return resolved


def import_from_workdir(module_name: str, workdir: str | Path) -> Any:
    ensure_import_path(workdir)
    return importlib.import_module(module_name)


def load_json_if_present(path: str | Path) -> dict[str, Any] | None:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_image(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape") or hasattr(value, "convert"):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return load_image(value[0]) if value else None
    if isinstance(value, (str, Path)):
        from PIL import Image

        return Image.open(value).convert("RGB")
    return value


def to_numpy_image(value: Any) -> Any:
    image = load_image(value)
    if image is None:
        return None
    import numpy as np

    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f" and array.max(initial=0) <= 1.0:
        array = (array * 255).astype("uint8")
    return array


def collect_images(observation: Mapping[str, Any], image: Any, keys: Sequence[str]) -> list[Any]:
    images = first_present(observation, "images")
    if isinstance(images, Mapping):
        return [images[key] for key in keys if key in images and images[key] is not None]
    if hasattr(images, "shape") or hasattr(images, "convert"):
        return [images]
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        return [item for item in images if item is not None]

    collected = [observation[key] for key in keys if key in observation and observation[key] is not None]
    if collected:
        return collected
    if image is None:
        return []
    if isinstance(image, Mapping):
        return [image[key] for key in keys if key in image and image[key] is not None]
    if hasattr(image, "shape") or hasattr(image, "convert"):
        return [image]
    if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
        return [item for item in image if item is not None]
    return [image]


def extract_action_values(value: Any) -> Any:
    if isinstance(value, Mapping) and "actions" in value:
        return value["actions"]
    if isinstance(value, list) and value and hasattr(value[0], "actions"):
        return [item.actions for item in value]
    if hasattr(value, "actions"):
        return value.actions
    return value


def completed_action_result(
    *,
    model_id: str,
    instruction: str,
    actions: Any,
    checkpoint_path: str,
    device: str,
    runtime: str,
    raw_output: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "model_id": model_id,
        "instruction": instruction,
        "actions": extract_action_values(actions),
        "checkpoint_path": checkpoint_path,
        "device": device,
        "runtime": runtime,
        "raw_output": raw_output if raw_output is not None else actions,
        "metadata": dict(metadata or {}),
    }
