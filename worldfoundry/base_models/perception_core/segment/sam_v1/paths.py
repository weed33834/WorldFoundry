"""Path helpers for SAM v1 checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


def checkpoint_path(model_type: str = "vit_b") -> Path:
    if model_type == "vit_b":
        explicit = os.environ.get("WORLDFOUNDRY_SAM_VIT_B_CKPT")
        filename = "sam_vit_b_01ec64.pth"
        fallback_dirs = ("", "evalcrafter", "sam")
    elif model_type == "vit_h":
        explicit = os.environ.get("WORLDFOUNDRY_SAM_VIT_H_CKPT")
        filename = "sam_vit_h_4b8939.pth"
        fallback_dirs = ("", "evalcrafter", "sam")
    else:
        raise ValueError(f"Unsupported SAM v1 model_type: {model_type}")

    if explicit:
        return Path(explicit).expanduser()

    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        for subdir in fallback_dirs:
            candidate = (root / subdir / filename) if subdir else root / filename
            if candidate.is_file():
                return candidate

    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "SAM" / filename
        if candidate.is_file():
            return candidate

    return Path("checkpoints") / "ckpt" / filename
