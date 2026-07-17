"""VideoX-Fun camera-control synthesis adapters."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .worldfoundry_runtime import (
    Wan21Fun1P3BCameraRuntime,
    Wan21Fun14BCameraRuntime,
    Wan22Fun5BCameraRuntime,
    Wan22FunA14BCameraRuntime,
)


class _WanFunCameraSynthesis(BaseSynthesis):
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


class Wan21Fun1P3BCameraSynthesis(_WanFunCameraSynthesis):
    RUNTIME_CLS = Wan21Fun1P3BCameraRuntime


class Wan21Fun14BCameraSynthesis(_WanFunCameraSynthesis):
    RUNTIME_CLS = Wan21Fun14BCameraRuntime


class Wan22Fun5BCameraSynthesis(_WanFunCameraSynthesis):
    RUNTIME_CLS = Wan22Fun5BCameraRuntime


class Wan22FunA14BCameraSynthesis(_WanFunCameraSynthesis):
    RUNTIME_CLS = Wan22FunA14BCameraRuntime


__all__ = [
    "Wan21Fun1P3BCameraSynthesis",
    "Wan21Fun14BCameraSynthesis",
    "Wan22Fun5BCameraSynthesis",
    "Wan22FunA14BCameraSynthesis",
]
