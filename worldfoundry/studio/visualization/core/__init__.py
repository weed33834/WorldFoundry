"""Backend-neutral visualization contracts and routing helpers."""

from __future__ import annotations

from .artifacts import StudioVisualizationArtifact, infer_visualization_artifact, normalize_artifact_uri
from .controls import VisualizationControl, VisualizationEvent
from .registry import (
    BackendCapabilities,
    RenderPlan,
    RenderRequest,
    RenderResult,
    StudioVisualizationBackend,
    StudioVisualizationEvent,
    StudioVisualizationLaunch,
    StudioVisualizationRegistry,
    StudioVisualizationRequest,
    VisualizationBackend,
    VisualizationProvider,
    model_visualization_profile,
    normalize_visualization_mode,
)
from .scene import Frame, Layer, Timeline, VisualizationScene
from .styles import VisualizationStyle

__all__ = [
    "BackendCapabilities",
    "Frame",
    "Layer",
    "RenderPlan",
    "RenderRequest",
    "RenderResult",
    "StudioVisualizationArtifact",
    "StudioVisualizationBackend",
    "StudioVisualizationEvent",
    "StudioVisualizationLaunch",
    "StudioVisualizationRegistry",
    "StudioVisualizationRequest",
    "Timeline",
    "VisualizationBackend",
    "VisualizationControl",
    "VisualizationEvent",
    "VisualizationProvider",
    "VisualizationScene",
    "VisualizationStyle",
    "infer_visualization_artifact",
    "model_visualization_profile",
    "normalize_artifact_uri",
    "normalize_visualization_mode",
]
