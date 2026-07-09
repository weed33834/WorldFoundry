"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> config.py functionality."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    """Checkpointing cfg implementation."""
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int


@dataclass
class ModelCfg:
    """Model cfg implementation."""
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    """Trainer cfg implementation."""
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None


@dataclass
class RootCfg:
    """Root cfg implementation."""
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    """Load typed config.

    Args:
        cfg: The cfg.
        data_class: The data class.
        extra_type_hooks: The extra type hooks.

    Returns:
        The return value.
    """
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    """Separate loss cfg wrappers.

    Args:
        joined: The joined.

    Returns:
        The return value.
    """
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        """Dummy implementation."""
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    """Load typed root config.

    Args:
        cfg: The cfg.

    Returns:
        The return value.
    """
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
