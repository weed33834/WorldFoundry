"""Path helpers for the VideoMAE action recognizer used by EvalCrafter.

MMAction2 is a framework dependency, so WorldFoundry only keeps the concrete
VideoMAEv2 recipe used by EvalCrafter here: one config, one label map, and the
expected checkpoint filename.
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_MAE_CHECKPOINT_NAME = "vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

_ROOT = Path(__file__).resolve().parent


def config_path() -> Path:
    return _ROOT / "configs" / "recognition" / "videomaev2" / (
        "vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py"
    )


def label_map_path() -> Path:
    return _ROOT / "labels" / "label_map_k400.txt"


def checkpoint_path(*, checkpoint_dir: str | os.PathLike[str] | None = None) -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_VIDEOMAE_K400_CKPT")
    if explicit:
        return Path(explicit).expanduser()

    roots: list[Path] = []
    if checkpoint_dir:
        roots.append(Path(checkpoint_dir).expanduser())
    env_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if env_dir:
        roots.append(Path(env_dir).expanduser() / "VideoMAE")
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        roots.extend(
            [
                Path(ckpt_dir).expanduser() / "VideoMAE",
                Path(ckpt_dir).expanduser() / "evalcrafter" / "VideoMAE",
            ]
        )

    for root in roots:
        candidate = root / VIDEO_MAE_CHECKPOINT_NAME
        if candidate.is_file():
            return candidate
    return (roots[0] if roots else Path("checkpoints") / "VideoMAE") / VIDEO_MAE_CHECKPOINT_NAME
