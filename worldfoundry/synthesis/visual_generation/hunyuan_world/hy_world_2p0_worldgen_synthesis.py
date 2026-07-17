"""HY-World 2.0 trajectory-render synthesis adapter."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .hy_world_2p0_worldgen_runtime import HYWorld2WorldgenRuntime


class HYWorld2WorldgenSynthesis(BaseSynthesis):
    def __init__(self, runtime: HYWorld2WorldgenRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "HYWorld2WorldgenSynthesis":
        return cls(HYWorld2WorldgenRuntime.from_pretrained(pretrained_model_path, **kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return self.runtime.predict(**kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.plan(**kwargs)


__all__ = ["HYWorld2WorldgenSynthesis"]
