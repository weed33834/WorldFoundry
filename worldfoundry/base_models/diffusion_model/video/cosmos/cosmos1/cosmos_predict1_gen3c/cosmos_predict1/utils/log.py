# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> utils -> log.py functionality."""

from __future__ import annotations

import atexit
import os
from typing import Any, Optional

import torch.distributed as dist
from loguru._logger import Core, Logger
from tqdm import tqdm

RANK0_ONLY = True
LEVEL = os.environ.get("LOGURU_LEVEL", "INFO")

logger = Logger(
    core=Core(),
    exception=None,
    depth=1,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

atexit.register(logger.remove)


def _add_relative_path(record: dict[str, Any]) -> None:
    """Helper function to add relative path.

    Args:
        record: The record.

    Returns:
        The return value.
    """
    start = os.getcwd()
    record["extra"]["relative_path"] = os.path.relpath(record["file"].path, start)


*options, _, extra = logger._options  # type: ignore
logger._options = tuple([*options, [_add_relative_path], extra])  # type: ignore


def init_loguru_stdout() -> None:
    """Init loguru stdout.

    Returns:
        The return value.
    """
    logger.remove()
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level=LEVEL,
        format=f"[<green>{{time:MM-DD HH:mm:ss}}</green>|{machine_format}{message_format}",
        filter=_rank0_only_filter,
    )


def init_loguru_file(path: str) -> None:
    """Init loguru file.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        path,
        encoding="utf8",
        level=LEVEL,
        format=f"[<green>{{time:MM-DD HH:mm:ss}}</green>|{machine_format}{message_format}",
        rotation="100 MB",
        filter=lambda result: _rank0_only_filter(result) or not RANK0_ONLY,
        enqueue=True,
    )


def get_machine_format() -> str:
    """Get machine format.

    Returns:
        The return value.
    """
    node_id = os.environ.get("NGC_ARRAY_INDEX", "0")
    num_nodes = int(os.environ.get("NGC_ARRAY_SIZE", "1"))
    machine_format = ""
    if dist.is_available() and not RANK0_ONLY and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        machine_format = (
            f"<red>[Node{node_id:<3}/{num_nodes:<3}][RANK{rank:<5}/{world_size:<5}][{{process.name:<8}}]</red>| "
        )
    return machine_format


def get_message_format() -> str:
    """Get message format.

    Returns:
        The return value.
    """
    return "<level>{level}</level>|<cyan>{extra[relative_path]}:{line}:{function}</cyan>] {message}"


def _rank0_only_filter(record: Any) -> bool:
    """Helper function to rank0 only filter.

    Args:
        record: The record.

    Returns:
        The return value.
    """
    is_rank0 = record["extra"].get("rank0_only", True)
    if _get_rank() == 0 and is_rank0:
        return True
    if not is_rank0:
        record["message"] = f"[RANK {_get_rank()}] " + record["message"]
    return not is_rank0


def trace(message: str, rank0_only: bool = True) -> None:
    """Trace.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).trace(message)


def debug(message: str, rank0_only: bool = True) -> None:
    """Debug.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).debug(message)


def info(message: str, rank0_only: bool = True) -> None:
    """Info.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).info(message)


def success(message: str, rank0_only: bool = True) -> None:
    """Success.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).success(message)


def warning(message: str, rank0_only: bool = True) -> None:
    """Warning.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).warning(message)


def error(message: str, rank0_only: bool = True) -> None:
    """Error.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).error(message)


def critical(message: str, rank0_only: bool = True) -> None:
    """Critical.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).critical(message)


def exception(message: str, rank0_only: bool = True) -> None:
    """Exception.

    Args:
        message: The message.
        rank0_only: The rank0 only.

    Returns:
        The return value.
    """
    logger.opt(depth=1).bind(rank0_only=rank0_only).exception(message)


def _get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Helper function to get rank.

    Args:
        group: The group.

    Returns:
        The return value.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


init_loguru_stdout()
