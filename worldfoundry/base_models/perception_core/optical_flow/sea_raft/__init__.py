"""Canonical SEA-RAFT optical-flow runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

from .core.parser import parse_args
from .core.raft import RAFT
from .core.utils.utils import load_ckpt


def config_path(name: str = "spring-M.json") -> Path:
    return Path(__file__).resolve().parent / "config" / "eval" / name


def checkpoint_path() -> Path:
    asset = BASE_MODEL_CAPABILITIES["sea_raft"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


__all__ = ["RAFT", "checkpoint_path", "config_path", "load_ckpt", "parse_args"]
