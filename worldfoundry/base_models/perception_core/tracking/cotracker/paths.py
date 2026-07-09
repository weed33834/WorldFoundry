"""Path helpers for the in-tree CoTracker runtime."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.io.paths import worldfoundry_path_tokens


def checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_COTRACKER2_CKPT")
    if explicit:
        return Path(explicit).expanduser()

    candidates: list[Path] = []
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR") or worldfoundry_path_tokens()["WORLDFOUNDRY_CKPT_DIR"]
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        candidates.extend(
            [
                root / "cotracker2.pth",
                root / "cotracker" / "cotracker2.pth",
                root / "co-tracker" / "cotracker2.pth",
                root / "VBench" / "cotracker2.pth",
                root / "VBench2" / "cotracker2.pth",
                root / "ChronoMagic-Bench" / "CHScore" / "cotracker2.pth",
                root / "BestWishYsh--ChronoMagic-Bench" / "CHScore" / "cotracker2.pth",
            ]
        )

    workspace_root = os.environ.get("WORLDFOUNDRY_WORKSPACE_ROOT")
    if workspace_root:
        root = Path(workspace_root).expanduser()
        candidates.extend(
            [
                root / "ckpt" / "cotracker2.pth",
                root / "ckpt" / "cotracker" / "cotracker2.pth",
                root / "ckpt" / "ChronoMagic-Bench" / "CHScore" / "cotracker2.pth",
                root / "ckpt" / "hfd" / "BestWishYsh--ChronoMagic-Bench" / "CHScore" / "cotracker2.pth",
            ]
        )

    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "cotracker2.pth")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]
