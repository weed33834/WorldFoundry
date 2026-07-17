"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> dataset -> view_sampler -> __init__.py functionality."""

from typing import Any

from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "evaluation": ViewSamplerEvaluation,
}

ViewSamplerCfg = ViewSamplerEvaluationCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
) -> ViewSampler[Any]:
    """Get view sampler.

    Args:
        cfg: The cfg.
        stage: The stage.
        overfit: The overfit.
        cameras_are_circular: The cameras are circular.

    Returns:
        The return value.
    """
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
    )
