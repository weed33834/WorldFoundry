# Minimal trajectory writer required by DUSt3R/MASt3R inference paths.
"""Module for base_models -> three_dimensions -> general_3d -> dust3r -> dust3r -> utils -> vo_eval.py functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_trajectory_tum_format(traj: Sequence[object], filename: str | Path) -> None:
    """Write a trajectory in TUM RGB-D text format.

    ``traj`` is expected to be ``(poses, timestamps)``, where poses are either
    ``N x 7`` ``x y z qx qy qz qw`` rows or ``N x 8`` rows that already include
    timestamps. This keeps inference-only runtime imports independent from the
    optional evo-based evaluation helpers in the upstream repositories.
    """
    poses = np.asarray(traj[0])
    timestamps = np.asarray(traj[1]).reshape(-1) if len(traj) > 1 else np.arange(len(poses), dtype=float)
    if poses.ndim != 2 or poses.shape[1] not in {7, 8}:
        raise ValueError(f"expected poses with shape N x 7 or N x 8, got {poses.shape}")
    rows = poses if poses.shape[1] == 8 else np.column_stack([timestamps[: len(poses)], poses])
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.9f")
