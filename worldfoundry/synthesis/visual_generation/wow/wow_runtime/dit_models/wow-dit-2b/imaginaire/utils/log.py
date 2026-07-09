# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight loguru facade for WoW inference.

The original helper imported ``torch.distributed`` and loguru at module import
time. That made config-only imports pay the full PyTorch startup cost. This
module keeps the same public logging functions while initializing heavy logging
dependencies only when a message is actually emitted.
"""

from __future__ import annotations

import atexit
import os
import sys
from typing import Any

RANK0_ONLY = True
LEVEL = os.environ.get("LOGURU_LEVEL", "INFO")

_LOGGER: Any | None = None
_LOGGER_REMOVE_REGISTERED = False
_STDOUT_INITIALIZED = False


class _LoggerProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(_get_logger(), name)


logger = _LoggerProxy()


def _get_logger() -> Any:
    global _LOGGER, _LOGGER_REMOVE_REGISTERED
    if _LOGGER is not None:
        return _LOGGER

    from loguru._logger import Core, Logger

    _LOGGER = Logger(
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
    if not _LOGGER_REMOVE_REGISTERED:
        atexit.register(_LOGGER.remove)
        _LOGGER_REMOVE_REGISTERED = True
    _patch_relative_path(_LOGGER)
    return _LOGGER


def _patch_relative_path(active_logger: Any) -> None:
    *options, _, extra = active_logger._options  # type: ignore[attr-defined]
    active_logger._options = tuple([*options, [_add_relative_path], extra])  # type: ignore[attr-defined]


def _add_relative_path(record: dict[str, Any]) -> None:
    start = os.getcwd()
    record["extra"]["relative_path"] = os.path.relpath(record["file"].path, start)


def init_loguru_stdout() -> None:
    global _STDOUT_INITIALIZED
    active_logger = _get_logger()
    active_logger.remove()
    machine_format = get_machine_format()
    message_format = get_message_format()
    active_logger.add(
        sys.stdout,
        level=LEVEL,
        format="[<green>{time:MM-DD HH:mm:ss}</green>|" f"{machine_format}" f"{message_format}",
        filter=_rank0_only_filter,
    )
    _STDOUT_INITIALIZED = True


def init_loguru_file(path: str) -> None:
    active_logger = _get_logger()
    machine_format = get_machine_format()
    message_format = get_message_format()
    active_logger.add(
        path,
        encoding="utf8",
        level=LEVEL,
        format="[<green>{time:MM-DD HH:mm:ss}</green>|" f"{machine_format}" f"{message_format}",
        rotation="100 MB",
        filter=lambda result: _rank0_only_filter(result) or not RANK0_ONLY,
        enqueue=True,
    )


def get_machine_format() -> str:
    if RANK0_ONLY:
        return ""
    rank = _get_rank()
    world_size = _get_world_size()
    if world_size <= 1:
        return ""
    node_id = os.environ.get("NODE_RANK", "0")
    num_nodes = os.environ.get("NNODES", "1")
    return f"<red>[Node{node_id:<3}/{num_nodes:<3}][RANK{rank:<5}/{world_size:<5}][{{process.name:<8}}]</red>| "


def get_message_format() -> str:
    return "<level>{level}</level>|<cyan>{extra[relative_path]}:{line}:{function}</cyan>] {message}"


def _ensure_stdout_logger() -> Any:
    if not _STDOUT_INITIALIZED:
        init_loguru_stdout()
    return _get_logger()


def _rank0_only_filter(record: Any) -> bool:
    is_rank0 = record["extra"].get("rank0_only", True)
    rank = _get_rank()
    if rank == 0 and is_rank0:
        return True
    if not is_rank0:
        record["message"] = f"[RANK{rank}] " + record["message"]
    return not is_rank0


def trace(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).trace(message)


def debug(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).debug(message)


def info(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).info(message)


def success(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).success(message)


def warning(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).warning(message)


def error(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).error(message)


def critical(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).critical(message)


def exception(message: str, rank0_only: bool = True) -> None:
    _ensure_stdout_logger().opt(depth=1).bind(rank0_only=rank0_only).exception(message)


def _get_rank(group: Any | None = None) -> int:
    for env_name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(env_name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass

    if _get_world_size_from_env() <= 1:
        return 0

    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return 0

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def _get_world_size() -> int:
    env_world_size = _get_world_size_from_env()
    if env_world_size > 1:
        return env_world_size

    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return 1

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_world_size_from_env() -> int:
    for env_name in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(env_name)
        if value is not None:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1
