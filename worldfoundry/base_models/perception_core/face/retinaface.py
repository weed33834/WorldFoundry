"""RetinaFace checkpoint helpers shared by benchmark runners."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import torch


def load_retinaface_state_dict(checkpoint: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load the ternaus RetinaFace checkpoint from a staged local file.

    The official checkpoint is distributed as a single-file zip archive. Benchmark
    code must not download it during scoring, so this helper only accepts local
    paths and raises a clear error when the asset is missing.
    """

    checkpoint_path = Path(checkpoint).expanduser()
    if str(checkpoint).startswith(("http://", "https://")):
        raise ValueError(
            "RetinaFace checkpoint must be staged locally; received a URL. "
            "Set WORLDFOUNDRY_VBENCH_RETINAFACE_CKPT or stage the VBench RetinaFace asset."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RetinaFace checkpoint is not staged: {checkpoint_path}")

    if zipfile.is_zipfile(checkpoint_path):
        with zipfile.ZipFile(checkpoint_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() in {".pth", ".pt", ".bin"}
            ]
            if not members:
                raise RuntimeError(f"RetinaFace checkpoint zip has no torch checkpoint file: {checkpoint_path}")
            with archive.open(members[0]) as handle:
                return torch.load(io.BytesIO(handle.read()), map_location=map_location)

    return torch.load(checkpoint_path, map_location=map_location)
