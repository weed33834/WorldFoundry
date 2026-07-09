"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> loss -> __init__.py functionality."""

import os

from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    """Get losses.

    Args:
        cfgs: The cfgs.

    Returns:
        The return value.
    """
    if os.environ.get("WORLDFOUNDRY_PIXELSPLAT_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}:
        cfgs = [cfg for cfg in cfgs if not isinstance(cfg, LossLpipsCfgWrapper)]
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
