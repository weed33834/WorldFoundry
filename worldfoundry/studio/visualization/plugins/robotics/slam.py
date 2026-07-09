"""Studio-side SLAM visualization hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DroidVisualizer:
    video: Any
    output_path: str | None = None

    def run(self) -> None:
        raise RuntimeError("DROID-SLAM viewer must be implemented in worldfoundry.studio, not base_models.")


def merge_depths_and_poses(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("DROID-SLAM visualization merge is not available in base_models.")


def visualization_fn(*args: Any, **kwargs: Any) -> None:
    DroidVisualizer(*args, **kwargs).run()


__all__ = ["DroidVisualizer", "merge_depths_and_poses", "visualization_fn"]
