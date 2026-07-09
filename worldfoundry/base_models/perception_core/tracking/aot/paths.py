"""Path helpers for the DeAOT checkpoint used by object tracking metrics."""

from __future__ import annotations

import os
from pathlib import Path


def checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_DEAOT_R50_CKPT")
    if explicit:
        return Path(explicit).expanduser()

    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "ckpt" / "R50_DeAOTL_PRE_YTB_DAV.pth"
        if candidate.is_file():
            return candidate

    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        roots = [
            root,
            root / "evalcrafter" / "ckpt",
            root / "Segment-and-Track-Anything" / "ckpt",
        ]
        for candidate_root in roots:
            candidate = candidate_root / "R50_DeAOTL_PRE_YTB_DAV.pth"
            if candidate.is_file():
                return candidate

    return Path("checkpoints") / "ckpt" / "R50_DeAOTL_PRE_YTB_DAV.pth"
