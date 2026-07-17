"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> dataset -> __init__.py functionality."""

from torch.utils.data import Dataset

from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
}


DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
) -> Dataset:
    """Get dataset.

    Args:
        cfg: The cfg.
        stage: The stage.
        step_tracker: The step tracker.

    Returns:
        The return value.
    """
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
