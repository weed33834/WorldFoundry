"""Path helpers for Keye-VL judge models."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "Kwai-Keye/Keye-VL-1_5-8B"


def _candidate_hfd_paths() -> list[Path]:
    paths: list[Path] = []
    hfd_root = os.environ.get("WORLDFOUNDRY_HFD_ROOT")
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if hfd_root:
        root = Path(hfd_root).expanduser()
        paths.extend(
            [
                root / "Kwai-Keye--Keye-VL-1_5-8B",
                root / "models--Kwai-Keye--Keye-VL-1_5-8B",
            ]
        )
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        paths.extend(
            [
                root / "hfd" / "Kwai-Keye--Keye-VL-1_5-8B",
                root / "Kwai-Keye--Keye-VL-1_5-8B",
                root / "Keye-VL-1_5-8B",
            ]
        )
    return paths


def model_path() -> str:
    """Return a local Keye-VL path when staged, otherwise the native HF repo id."""
    explicit = (
        os.environ.get("WORLDFOUNDRY_KEYE_VL_MODEL")
        or os.environ.get("WORLDFOUNDRY_4DWORLDBENCH_KEYE_MODEL")
        or os.environ.get("KEYE_MODEL_PATH")
    )
    if explicit:
        return explicit
    for candidate in _candidate_hfd_paths():
        if candidate.exists():
            return str(candidate)
    return HF_REPO_ID
