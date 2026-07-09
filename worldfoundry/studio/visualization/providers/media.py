"""Media artifact provider for Studio visualization scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene

MEDIA_KINDS = {"image", "video", "audio"}


class MediaProvider:
    provider_id = "media"

    def discover(self, source) -> VisualizationScene | None:
        paths = _source_paths(source)
        layers = []
        for path in paths:
            artifact = infer_visualization_artifact(path)
            if artifact.kind not in MEDIA_KINDS:
                continue
            layers.append(Layer(layer_id=path.stem, kind=artifact.kind, uri=path.as_posix(), metadata={"format": artifact.format_hint}))
        if not layers:
            return None
        return VisualizationScene(scene_id=f"media/{Path(paths[0]).stem}", title="Media Preview", layers=tuple(layers), recommended_backend="media")


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
