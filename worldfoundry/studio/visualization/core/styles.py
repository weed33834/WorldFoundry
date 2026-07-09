"""Backend-neutral style helpers for visualization layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULT_COLORMAP = 'viridis'


@dataclass(frozen=True)
class VisualizationStyle:
    color: str | tuple[float, float, float] | tuple[float, float, float, float] | None = None
    colormap: str = DEFAULT_COLORMAP
    opacity: float | None = None
    point_size: float | None = None
    line_width: float | None = None
    material: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                'color': self.color,
                'colormap': self.colormap,
                'opacity': self.opacity,
                'point_size': self.point_size,
                'line_width': self.line_width,
                'material': self.material,
                'metadata': dict(self.metadata),
            }.items()
            if value not in (None, {})
        }
