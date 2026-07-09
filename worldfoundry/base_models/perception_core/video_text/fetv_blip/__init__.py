"""FETV BLIP retrieval source used by in-tree evaluation runtimes."""

from __future__ import annotations

from pathlib import Path

FETV_BLIP_ROOT = Path(__file__).resolve().parent
BLIP_CONFIG_PATH = FETV_BLIP_ROOT / "blip_config.yaml"
MED_CONFIG_PATH = FETV_BLIP_ROOT / "configs" / "med_config.json"

__all__ = ["BLIP_CONFIG_PATH", "FETV_BLIP_ROOT", "MED_CONFIG_PATH"]
