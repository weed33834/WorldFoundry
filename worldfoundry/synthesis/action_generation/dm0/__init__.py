"""Inference-only, in-tree DM0 integration with lazy heavy imports."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"DM0Config", "DM0ForCausalLM"}:
        from .modeling import DM0Config, DM0ForCausalLM

        return {"DM0Config": DM0Config, "DM0ForCausalLM": DM0ForCausalLM}[name]
    if name in {"DM0Runtime", "DM0RuntimeConfig", "predict_action"}:
        from .runtime import DM0Runtime, DM0RuntimeConfig, predict_action

        return {
            "DM0Runtime": DM0Runtime,
            "DM0RuntimeConfig": DM0RuntimeConfig,
            "predict_action": predict_action,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DM0Config",
    "DM0ForCausalLM",
    "DM0Runtime",
    "DM0RuntimeConfig",
    "predict_action",
]
