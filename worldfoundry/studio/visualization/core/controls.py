"""Normalized Studio visualization controls and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VisualizationControl:
    control_id: str
    kind: str
    label: str = ''
    value: Any | None = None
    options: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationEvent:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
