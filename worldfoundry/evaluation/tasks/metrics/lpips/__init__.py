"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .compute import NetType, compute_lpips

METRIC_ID = "lpips"
ALIASES: tuple[str, ...] = ()
HIGHER_IS_BETTER = False
FAMILY = "perceptual"
TAGS = ("perceptual", "condition_consistency")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Learned Perceptual Image Patch Similarity (pairwise, lower is better).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_lpips

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "NetType", "compute", "compute_lpips"]
