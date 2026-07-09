"""Path helpers for the canonical FlowFormer++ runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES


def checkpoint_path() -> Path:
    """Resolve the FlowFormer++ Things checkpoint through the capability registry."""
    asset = BASE_MODEL_CAPABILITIES["flowformerplusplus"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


__all__ = ["checkpoint_path"]
