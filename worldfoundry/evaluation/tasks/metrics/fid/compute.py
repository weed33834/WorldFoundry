"""FID computation via in-tree torch-fidelity (Inception-v3 features)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics


def resolve_distribution_inputs(
    reference: str | Path | Sequence[str | Path] | None,
    generated: str | Path | Sequence[str | Path] | None,
) -> tuple[str | Path, str | Path]:
    if reference is None or generated is None:
        raise ValueError("reference and generated inputs are required")
    return reference, generated


def _parse_numeric_result(result: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_distribution_metrics(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    metrics: Sequence[str] = ("fid",),
    feature_extractor: str = "inception-v3-compat",
    batch_size: int = 64,
    cuda: bool = True,
    **kwargs: Any,
) -> dict[str, float]:
    """Run torch-fidelity distribution metrics (backward-compat helper; prefer metric-local APIs)."""
    ref, gen = resolve_distribution_inputs(reference, generated)
    metric_flags = {name: True for name in metrics}
    result = calculate_metrics()(
        input1=str(ref),
        input2=str(gen),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        **metric_flags,
        **kwargs,
    )
    return _parse_numeric_result(result)


_SWAV_EXTRACTORS = frozenset({"swav", "swav-resnet50", "swav_resnet50"})


def compute_fid(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> float:
    if feature_extractor in _SWAV_EXTRACTORS:
        from .swav import compute_swav_fid

        device = kwargs.pop("device", None)
        if device is None and not cuda:
            device = "cpu"
        return compute_swav_fid(
            reference,
            generated,
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
    result = calculate_metrics()(
        input1=str(reference),
        input2=str(generated),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        fid=True,
        **kwargs,
    )
    parsed = _parse_numeric_result(result)
    for key, value in parsed.items():
        if "fid" in key.lower():
            return value
    raise KeyError("FID key missing from torch-fidelity result")


def summarize_distribution_metrics(payload: Mapping[str, float]) -> dict[str, float]:
    """Normalize torch-fidelity metric keys to stable WorldFoundry names."""
    aliases = {
        "inception_score_mean": "is_mean",
        "inception_score_std": "is_std",
        "frechet_inception_distance": "fid",
        "kernel_inception_distance_mean": "kid_mean",
        "kernel_inception_distance_std": "kid_std",
        "monge_inception_distance": "mind",
        "perceptual_path_length_mean": "ppl_mean",
        "perceptual_path_length_std": "ppl_std",
        "perceptual_path_length_raw": "ppl_raw",
        "precision": "precision",
        "recall": "recall",
        "f_score": "f_score",
    }
    summary: dict[str, float] = {}
    for key, value in payload.items():
        normalized = aliases.get(key, key)
        summary[normalized] = float(value)
    return summary


__all__ = [
    "compute_distribution_metrics",
    "compute_fid",
    "resolve_distribution_inputs",
    "summarize_distribution_metrics",
]
