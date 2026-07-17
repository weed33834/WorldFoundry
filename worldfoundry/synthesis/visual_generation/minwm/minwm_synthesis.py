"""minWM synthesis adapters."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .worldfoundry_runtime import MinWMHYAction2VRuntime, MinWMWanAction2VRuntime


class _MinWMSynthesis(BaseSynthesis):
    RUNTIME_CLS = None

    def __init__(self, runtime: Any) -> None:
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any):
        return cls(cls.RUNTIME_CLS.from_pretrained(pretrained_model_path, **kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return self.runtime.predict(**kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.plan(**kwargs)


class MinWMHYAction2VSynthesis(_MinWMSynthesis):
    RUNTIME_CLS = MinWMHYAction2VRuntime


class MinWMWanAction2VSynthesis(_MinWMSynthesis):
    RUNTIME_CLS = MinWMWanAction2VRuntime


__all__ = ["MinWMHYAction2VSynthesis", "MinWMWanAction2VSynthesis"]
