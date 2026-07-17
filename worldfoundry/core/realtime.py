"""Framework contracts for resident interactive world models.

The browser transport and Studio scheduler must not know a model's latent
stride, causal seed cadence, or preferred control vocabulary.  A model-owned
resident session reports those details through :class:`RealtimeSpec` while
keeping the hot-path API deliberately small: configure once, advance one
chunk, and reset without unloading weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


DEFAULT_REALTIME_CONTROLS = (
    "forward",
    "backward",
    "left",
    "right",
    "camera_up",
    "camera_down",
    "camera_l",
    "camera_r",
)


@dataclass(frozen=True, slots=True)
class RealtimeSpec:
    """Model-owned playback and generation cadence for one resident session."""

    fps: int = 16
    first_chunk_frames: int = 9
    steady_chunk_frames: int = 9
    controls: tuple[str, ...] = DEFAULT_REALTIME_CONTROLS
    transport: str = "in-memory-rgb"
    stateful: bool = True

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ValueError("RealtimeSpec.fps must be positive.")
        if self.first_chunk_frames < 1 or self.steady_chunk_frames < 1:
            raise ValueError("RealtimeSpec chunk frame counts must be positive.")
        if not self.controls:
            raise ValueError("RealtimeSpec.controls must not be empty.")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["controls"] = list(self.controls)
        return payload

    @classmethod
    def from_payload(
        cls,
        value: Any,
        *,
        fallback: "RealtimeSpec | None" = None,
    ) -> "RealtimeSpec":
        """Parse a model result without letting malformed metadata break play."""

        default = fallback or cls()
        if isinstance(value, Mapping) and isinstance(value.get("realtime_spec"), Mapping):
            value = value["realtime_spec"]
        if not isinstance(value, Mapping):
            return default
        try:
            controls = value.get("controls", default.controls)
            if isinstance(controls, str):
                controls = tuple(item.strip() for item in controls.split(",") if item.strip())
            else:
                controls = tuple(str(item) for item in controls)
            return cls(
                fps=int(value.get("fps", default.fps)),
                first_chunk_frames=int(
                    value.get("first_chunk_frames", default.first_chunk_frames)
                ),
                steady_chunk_frames=int(
                    value.get("steady_chunk_frames", default.steady_chunk_frames)
                ),
                controls=controls or default.controls,
                transport=str(value.get("transport", default.transport)),
                stateful=bool(value.get("stateful", default.stateful)),
            )
        except (TypeError, ValueError):
            return default


@runtime_checkable
class InteractiveWorldPipeline(Protocol):
    """Structural API implemented by model-specific resident adapters."""

    def prepare_realtime(self) -> Mapping[str, Any] | None: ...

    def configure_realtime(self, images: Any, prompt: str = "", **kwargs: Any) -> Any: ...

    def stream_realtime(self, interactions: list[str], **kwargs: Any) -> Any: ...

    def reset_realtime(self) -> None: ...


__all__ = [
    "DEFAULT_REALTIME_CONTROLS",
    "InteractiveWorldPipeline",
    "RealtimeSpec",
]
