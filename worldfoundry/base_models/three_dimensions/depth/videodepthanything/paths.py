"""Checkpoint discovery for the shared Video-Depth-Anything runtime."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.evaluation.utils import REPO_ROOT


def small_checkpoint_path() -> Path | None:
    explicit = os.environ.get("WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT")
    if explicit:
        return Path(explicit).expanduser()
    roots: list[Path] = []
    checkpoint_root = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if checkpoint_root:
        roots.append(Path(checkpoint_root).expanduser())
    roots.extend([REPO_ROOT / "checkpoints", REPO_ROOT.parent / "ckpt"])
    for root in roots:
        for relative in (
            Path("Video-Depth-Anything-Small") / "video_depth_anything_vits.pth",
            Path("Video-Depth-Anything") / "video_depth_anything_vits.pth",
            Path("video_depth_anything") / "video_depth_anything_vits.pth",
        ):
            candidate = root / relative
            if candidate.is_file():
                return candidate
    return None
