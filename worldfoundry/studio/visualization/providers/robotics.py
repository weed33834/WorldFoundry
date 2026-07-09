"""Robotics artifact provider for action traces and simulator replays."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene

TRACE_SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".pkl"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


class RoboticsProvider:
    provider_id = "robotics"

    def discover(self, source) -> VisualizationScene | None:
        layers = []
        for path in _source_paths(source):
            lowered = path.name.lower()
            if path.suffix.lower() in TRACE_SUFFIXES and any(token in lowered for token in ("action", "policy", "robot", "trajectory", "rollout")):
                layers.append(Layer(layer_id=path.stem, kind="action_trace", uri=path.as_posix()))
            elif path.suffix.lower() in VIDEO_SUFFIXES and any(token in lowered for token in ("sim", "robot", "rollout", "episode", "replay")):
                layers.append(Layer(layer_id=path.stem, kind="video", uri=path.as_posix(), metadata={"role": "simulator_replay"}))
        if not layers:
            return None
        return VisualizationScene(scene_id="robotics/artifacts", title="Robotics Artifacts", layers=tuple(layers), recommended_backend="embodied")


def _source_paths(source) -> list[Path]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return sorted(item for item in path.rglob("*") if item.is_file())
        return [path]
    if isinstance(source, Iterable):
        return [Path(item) for item in source]
    return []
