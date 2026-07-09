"""In-tree VQAScore backend (vendored t2v_metrics VQA path)."""

from __future__ import annotations

import os
import shutil
import subprocess

_FFMPEG_CHECKED = False


def ensure_ffmpeg() -> None:
    """Validate ffmpeg availability before video scoring paths run."""
    global _FFMPEG_CHECKED
    if _FFMPEG_CHECKED:
        return
    try:
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "ffmpeg is a required system requirement but not found. Install with:\n"
            "conda install ffmpeg=6.1.2 -c conda-forge\n"
            "or visit: https://ffmpeg.org/download.html"
        ) from exc
    _FFMPEG_CHECKED = True


__all__ = ["ensure_ffmpeg"]
