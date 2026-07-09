from __future__ import annotations

from typing import Any

from worldfoundry.base_models.diffusion_model.image.sana.worldfoundry_runtime import SanaRuntime

from ...base_synthesis import BaseSynthesis


class SanaSynthesis(BaseSynthesis):
    """Thin synthesis facade over the Sana runtime."""

    MODEL_ID = SanaRuntime.MODEL_ID
    DISPLAY_NAME = SanaRuntime.DISPLAY_NAME

    def __init__(self, *, runtime: SanaRuntime | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.runtime = runtime or SanaRuntime(**kwargs)

    def __getattr__(self, name: str):
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "SanaSynthesis":
        return cls(runtime=SanaRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.predict(*args, **kwargs)


__all__ = ["SanaSynthesis"]
