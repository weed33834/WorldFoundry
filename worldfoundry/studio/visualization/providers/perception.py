"""Perception artifact provider for masks, boxes, tracks, depth, and flow outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene

NAME_KIND_HINTS = {
    "mask": "segmentation",
    "segmentation": "segmentation",
    "box": "detection",
    "bbox": "detection",
    "keypoint": "keypoints",
    "track": "trajectory",
    "flow": "optical_flow",
    "depth": "depth",
}


class PerceptionProvider:
    provider_id = "perception"

    def discover(self, source) -> VisualizationScene | None:
        layers = []
        for path in _source_paths(source):
            kind = _kind_from_name(path) or infer_visualization_artifact(path).kind
            if kind in {"artifact", "audio"}:
                continue
            layers.append(Layer(layer_id=path.stem, kind=kind, uri=path.as_posix()))
        if not layers:
            return None
        return VisualizationScene(
            scene_id="perception/artifacts",
            title="Perception Artifacts",
            layers=tuple(layers),
            recommended_backend="media",
        )


def _kind_from_name(path: Path) -> str | None:
    lowered = path.name.lower()
    for token, kind in NAME_KIND_HINTS.items():
        if token in lowered:
            return kind
    return None


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
