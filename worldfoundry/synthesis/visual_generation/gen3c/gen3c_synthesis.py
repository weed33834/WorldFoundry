from __future__ import annotations

from typing import Any

from worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c import Gen3CRuntime

from ...base_synthesis import BaseSynthesis


class Gen3CSynthesis(BaseSynthesis):
    """Thin synthesis facade over the base-model GEN3C runtime."""

    def __init__(self, *, runtime: Gen3CRuntime | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.runtime = runtime or Gen3CRuntime(**kwargs)

    def __getattr__(self, name: str):
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Gen3CSynthesis":
        return cls(runtime=Gen3CRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any):
        return self.runtime.predict(*args, **kwargs)


__all__ = ["Gen3CSynthesis"]
