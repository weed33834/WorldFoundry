from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis
from .worldfoundry_runtime import ShowORuntime


class ShowOSynthesis(BaseSynthesis):
    """Thin synthesis facade over the Show-O runtime."""

    MODEL_ID = ShowORuntime.MODEL_ID
    DISPLAY_NAME = ShowORuntime.DISPLAY_NAME

    def __init__(self, *, runtime: ShowORuntime | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.runtime = runtime or ShowORuntime(**kwargs)

    def __getattr__(self, name: str):
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "ShowOSynthesis":
        return cls(runtime=ShowORuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.predict(*args, **kwargs)


__all__ = ["ShowOSynthesis"]
