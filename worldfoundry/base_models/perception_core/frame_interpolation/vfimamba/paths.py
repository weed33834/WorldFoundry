"""Path helpers for the canonical VFIMamba frame-interpolation runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES


def checkpoint_path(name: str = "VFIMamba") -> Path:
    """Resolve a VFIMamba checkpoint by model name through the capability registry."""
    capability = BASE_MODEL_CAPABILITIES["vfimamba"]
    target_id = f"vfimamba_{name.lower()}_checkpoint"
    for asset in capability.assets:
        if asset.id == target_id:
            status = asset.check()
            return Path(status["matched_path"] or status["local_path"])
    available = ", ".join(asset.id for asset in capability.assets)
    raise KeyError(f"unknown VFIMamba checkpoint {name!r}; available assets: {available}")


__all__ = ["checkpoint_path"]
