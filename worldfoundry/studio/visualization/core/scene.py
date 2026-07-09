"""Backend-neutral Studio visualization scene model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

LayerKind = str


@dataclass(frozen=True)
class Frame:
    """One named coordinate frame in a visualization scene."""

    frame_id: str
    parent_id: str | None = None
    transform: Sequence[Sequence[float]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Timeline:
    """Optional temporal metadata shared by time-varying layers."""

    fps: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Layer:
    """Backend-neutral layer entry consumed by providers and renderers."""

    layer_id: str
    kind: LayerKind
    uri: str | None = None
    uris: tuple[str, ...] = ()
    payload: Any | None = None
    frame_range: tuple[int, int] | None = None
    time_range: tuple[float, float] | None = None
    coordinate_frame: str | None = None
    style: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def all_uris(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.uri:
            values.append(self.uri)
        values.extend(value for value in self.uris if value)
        return tuple(dict.fromkeys(values))

    def asdict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            'layer_id': self.layer_id,
            'kind': self.kind,
        }
        if self.uri is not None:
            data['uri'] = self.uri
        if self.uris:
            data['uris'] = list(self.uris)
        if self.frame_range is not None:
            data['frame_range'] = list(self.frame_range)
        if self.time_range is not None:
            data['time_range'] = list(self.time_range)
        if self.coordinate_frame is not None:
            data['coordinate_frame'] = self.coordinate_frame
        if self.style:
            data['style'] = dict(self.style)
        if self.metadata:
            data['metadata'] = dict(self.metadata)
        return data

    @classmethod
    def fromdict(cls, data: Mapping[str, Any]) -> 'Layer':
        frame_range = data.get('frame_range')
        time_range = data.get('time_range')
        return cls(
            layer_id=str(data.get('layer_id') or data.get('id') or ''),
            kind=str(data.get('kind') or ''),
            uri=str(data['uri']) if data.get('uri') is not None else None,
            uris=tuple(str(item) for item in data.get('uris') or ()),
            frame_range=tuple(frame_range) if frame_range is not None else None,
            time_range=tuple(time_range) if time_range is not None else None,
            coordinate_frame=str(data['coordinate_frame']) if data.get('coordinate_frame') is not None else None,
            style=dict(data.get('style') or {}),
            metadata=dict(data.get('metadata') or {}),
        )


@dataclass(frozen=True)
class VisualizationScene:
    """Single source of truth passed from providers to visualization backends."""

    scene_id: str
    title: str = ''
    layers: tuple[Layer, ...] = ()
    timeline: Timeline | None = None
    controls: tuple[Any, ...] = ()
    frames: tuple[Frame, ...] = ()
    recommended_backend: str = 'auto'
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def layer_kinds(self) -> frozenset[str]:
        return frozenset(layer.kind for layer in self.layers)

    def asdict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            'schema_version': 1,
            'scene_id': self.scene_id,
            'title': self.title,
            'recommended_backend': self.recommended_backend,
            'layers': [layer.asdict() for layer in self.layers],
        }
        if self.timeline is not None:
            data['timeline'] = {
                key: value
                for key, value in {
                    'fps': self.timeline.fps,
                    'start_time': self.timeline.start_time,
                    'end_time': self.timeline.end_time,
                    'frame_count': self.timeline.frame_count,
                    'metadata': dict(self.timeline.metadata),
                }.items()
                if value not in (None, {})
            }
        if self.metadata:
            data['metadata'] = dict(self.metadata)
        if self.frames:
            data['frames'] = [
                {
                    'frame_id': frame.frame_id,
                    'parent_id': frame.parent_id,
                    'transform': frame.transform,
                    'metadata': dict(frame.metadata),
                }
                for frame in self.frames
            ]
        return data

    @classmethod
    def fromdict(cls, data: Mapping[str, Any]) -> 'VisualizationScene':
        timeline_blob = data.get('timeline')
        timeline = None
        if isinstance(timeline_blob, Mapping):
            timeline = Timeline(
                fps=timeline_blob.get('fps'),
                start_time=timeline_blob.get('start_time'),
                end_time=timeline_blob.get('end_time'),
                frame_count=timeline_blob.get('frame_count'),
                metadata=dict(timeline_blob.get('metadata') or {}),
            )
        return cls(
            scene_id=str(data.get('scene_id') or ''),
            title=str(data.get('title') or ''),
            recommended_backend=str(data.get('recommended_backend') or 'auto'),
            layers=tuple(Layer.fromdict(item) for item in data.get('layers') or ()),
            timeline=timeline,
            metadata=dict(data.get('metadata') or {}),
        )
