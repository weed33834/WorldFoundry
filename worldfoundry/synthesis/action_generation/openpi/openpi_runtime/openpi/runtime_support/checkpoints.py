from __future__ import annotations

import logging
from pathlib import Path

import openpi.shared.normalize as _normalize


def load_norm_stats(assets_dir: Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info("Loaded norm stats from %s", norm_stats_dir)
    return norm_stats


def _training_not_packaged(*args, **kwargs):
    raise RuntimeError("OpenPI training checkpoint save/restore is not packaged in WorldFoundry.")


initialize_checkpoint_dir = _training_not_packaged
save_state = _training_not_packaged
restore_state = _training_not_packaged
