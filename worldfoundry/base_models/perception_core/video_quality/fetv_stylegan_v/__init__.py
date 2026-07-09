"""FETV StyleGAN-V FVD source used by in-tree evaluation runtimes."""

from __future__ import annotations

from pathlib import Path

FETV_STYLEGAN_V_ROOT = Path(__file__).resolve().parent
CALC_METRICS_SCRIPT = FETV_STYLEGAN_V_ROOT / "src" / "scripts" / "calc_metrics_for_dataset.py"

__all__ = ["CALC_METRICS_SCRIPT", "FETV_STYLEGAN_V_ROOT"]
