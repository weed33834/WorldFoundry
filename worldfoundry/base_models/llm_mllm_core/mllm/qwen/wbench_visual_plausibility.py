"""Qwen3-VL PAVRM asset path used by WBench visual plausibility."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES


def model_dir() -> Path:
    asset = BASE_MODEL_CAPABILITIES["wbench_pavrm_qwen3vl"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


__all__ = ["model_dir"]
