"""Rank-aware logging helpers for distributed runtime code."""

from __future__ import annotations

import logging

import torch


class DistributedLogger:
    _logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls, *, level: int = logging.INFO) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("worldfoundry_distributed")
            cls._logger.setLevel(level)
            cls._logger.propagate = False
            cls._logger.handlers.clear()
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)
        return cls._logger


distributed_logger = DistributedLogger.get_logger()


def print_per_rank(message: object) -> None:
    distributed_logger.info(message)


def print_rank_0(message: object) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            distributed_logger.info(message)
    else:
        distributed_logger.info(message)


__all__ = ["DistributedLogger", "distributed_logger", "print_per_rank", "print_rank_0"]
