"""Path helpers for the FAST-VQA/FasterVQA video-quality runtime."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_CHECKPOINT_FILENAMES = {
    "FasterVQA": "FAST_VQA_3D_1_1_Scr.pth",
    "FasterVQA-MS": "FAST_VQA_3D_1_1_Scr.pth",
    "FasterVQA-MT": "FAST_VQA_3D_1_1_Scr.pth",
    "FAST-VQA": "FAST_VQA_B_1_4.pth",
    "FAST-VQA-M": "FAST_VQA_M_1_4.pth",
}


def runtime_root() -> Path:
    """Return the import root that contains the vendored `fastvqa` package."""
    return _ROOT


def options_dir() -> Path:
    """Return the official FAST-VQA option directory used by 4DWorldBench."""
    return _ROOT / "options" / "fast"


def _candidate_paths(filename: str) -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("WORLDFOUNDRY_FASTVQA_CKPT", "WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT"):
        explicit = os.environ.get(env_name)
        if explicit:
            candidates.append(Path(explicit).expanduser())
    for env_name in ("WORLDFOUNDRY_FASTVQA_CKPT_DIR", "WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT_DIR"):
        explicit_dir = os.environ.get(env_name)
        if explicit_dir:
            candidates.append(Path(explicit_dir).expanduser() / filename)
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        candidates.extend(
            [
                root / "FAST-VQA" / filename,
                root / "fastvqa" / filename,
                root / "4dworldbench" / "FAST-VQA" / filename,
                root / "WorldScore" / "metrics" / "checkpoints" / filename,
            ]
        )
    return candidates


def checkpoint_path(model_name: str = "FasterVQA") -> str:
    """Resolve the checkpoint path for a FAST-VQA/FasterVQA model variant."""
    filename = _CHECKPOINT_FILENAMES.get(model_name, _CHECKPOINT_FILENAMES["FasterVQA"])
    for candidate in _candidate_paths(filename):
        if candidate.exists():
            return str(candidate)
    return str(_candidate_paths(filename)[0]) if _candidate_paths(filename) else filename
