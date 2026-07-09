from __future__ import annotations

from typing import Any

from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat_runtime import PixelSplatRuntime

from ...base_synthesis import BaseSynthesis


class PixelSplatSynthesis(BaseSynthesis):
    """Thin synthesis facade over the base-model pixelSplat runtime."""

    MODEL_ID = PixelSplatRuntime.MODEL_ID
    DISPLAY_NAME = PixelSplatRuntime.DISPLAY_NAME

    def __init__(self, *, runtime: PixelSplatRuntime | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.runtime = runtime or PixelSplatRuntime(**kwargs)

    def __getattr__(self, name: str):
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "PixelSplatSynthesis":
        return cls(runtime=PixelSplatRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.predict(*args, **kwargs)


__all__ = ["PixelSplatSynthesis"]
