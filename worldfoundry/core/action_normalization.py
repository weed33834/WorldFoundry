"""Model-agnostic normalization helpers for continuous robot actions.

The functions in this module intentionally depend only on NumPy.  Policy
integrations can therefore share the checkpoint-statistics contract without
copying model-specific post-processing code or importing a training stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_BOUNDED_MODES = {
    "min_max": ("min", "max"),
    "q99": ("q01", "q99"),
    "quantile": ("q01", "q99"),
    "bounds": ("q01", "q99"),
}


def select_modality_statistics(
    dataset_statistics: Mapping[str, Any],
    *,
    modality: str = "action",
    key: str | None = None,
) -> tuple[str | None, Mapping[str, Any]]:
    """Select one modality's statistics from a checkpoint statistics mapping.

    Both common layouts are accepted: a direct ``{"min": ..., "max": ...}``
    mapping and a dataset-keyed ``{"robot": {"action": {...}}}`` mapping.
    Dataset-key selection is strict when more than one key is available.
    """

    if not isinstance(dataset_statistics, Mapping) or not dataset_statistics:
        raise ValueError("Dataset normalization statistics are empty or invalid.")

    if modality in dataset_statistics and isinstance(dataset_statistics[modality], Mapping):
        return key, dataset_statistics[modality]
    if any(name in dataset_statistics for name in ("min", "max", "q01", "q99", "mean", "std")):
        return key, dataset_statistics

    available = tuple(str(item) for item in dataset_statistics)
    if key is None:
        if len(available) != 1:
            raise ValueError(
                f"Multiple normalization keys are available: {list(available)}; "
                "select one explicitly."
            )
        key = available[0]
    if key not in dataset_statistics:
        raise KeyError(f"Normalization key {key!r} is unavailable; choices: {list(available)}.")

    selected = dataset_statistics[key]
    if not isinstance(selected, Mapping) or not isinstance(selected.get(modality), Mapping):
        raise KeyError(f"Normalization key {key!r} has no {modality!r} statistics.")
    return key, selected[modality]


def _coerce_values_and_mask(
    values: Any,
    statistics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("Action values must have at least one dimension.")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    else:
        array = array.copy()

    width = array.shape[-1]
    raw_mask = statistics.get("mask")
    mask = np.ones(width, dtype=bool) if raw_mask is None else np.asarray(raw_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != width:
        raise ValueError(f"Normalization mask has shape {mask.shape}; expected ({width},).")
    return array, mask


def _stat_vector(statistics: Mapping[str, Any], name: str, width: int, dtype: np.dtype) -> np.ndarray:
    if name not in statistics:
        raise KeyError(f"Normalization statistics are missing {name!r}.")
    vector = np.asarray(statistics[name], dtype=dtype)
    if vector.ndim != 1 or vector.shape[0] != width:
        raise ValueError(f"Normalization statistic {name!r} has shape {vector.shape}; expected ({width},).")
    return vector


def normalize_action_values(
    values: Any,
    statistics: Mapping[str, Any],
    *,
    mode: str = "min_max",
    clip: float | None = None,
) -> np.ndarray:
    """Normalize action values using checkpoint statistics along the last axis."""

    normalized, mask = _coerce_values_and_mask(values, statistics)
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode in _BOUNDED_MODES:
        low_name, high_name = _BOUNDED_MODES[normalized_mode]
        low = _stat_vector(statistics, low_name, normalized.shape[-1], normalized.dtype)
        high = _stat_vector(statistics, high_name, normalized.shape[-1], normalized.dtype)
        valid = mask & (high != low)
        normalized[..., valid] = 2.0 * (
            (normalized[..., valid] - low[valid]) / (high[valid] - low[valid])
        ) - 1.0
    elif normalized_mode in {"mean_std", "standard", "zscore"}:
        mean = _stat_vector(statistics, "mean", normalized.shape[-1], normalized.dtype)
        std = _stat_vector(statistics, "std", normalized.shape[-1], normalized.dtype)
        valid = mask & (std != 0)
        normalized[..., valid] = (normalized[..., valid] - mean[valid]) / std[valid]
    elif normalized_mode in {"identity", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported action normalization mode: {mode!r}.")
    if clip is not None:
        normalized[..., mask] = np.clip(normalized[..., mask], -float(clip), float(clip))
    return normalized


def unnormalize_action_values(
    normalized_values: Any,
    statistics: Mapping[str, Any],
    *,
    mode: str = "min_max",
) -> np.ndarray:
    """Convert normalized policy outputs back to environment-space actions."""

    actions, mask = _coerce_values_and_mask(normalized_values, statistics)
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode in _BOUNDED_MODES:
        low_name, high_name = _BOUNDED_MODES[normalized_mode]
        low = _stat_vector(statistics, low_name, actions.shape[-1], actions.dtype)
        high = _stat_vector(statistics, high_name, actions.shape[-1], actions.dtype)
        actions[..., mask] = (
            (actions[..., mask] + 1.0) * 0.5 * (high[mask] - low[mask]) + low[mask]
        )
    elif normalized_mode in {"mean_std", "standard", "zscore"}:
        mean = _stat_vector(statistics, "mean", actions.shape[-1], actions.dtype)
        std = _stat_vector(statistics, "std", actions.shape[-1], actions.dtype)
        actions[..., mask] = actions[..., mask] * std[mask] + mean[mask]
    elif normalized_mode in {"identity", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported action normalization mode: {mode!r}.")
    return actions


__all__ = [
    "normalize_action_values",
    "select_modality_statistics",
    "unnormalize_action_values",
]
