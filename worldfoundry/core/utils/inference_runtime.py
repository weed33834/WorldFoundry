"""Tensor-tree and deterministic random helpers used during inference."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from worldfoundry.core.io.termcolor import Color

from .torch_utils import set_random_seed


def to(data: Any, *, device=None, dtype=None, memory_format=torch.preserve_format) -> Any:
    if isinstance(data, torch.Tensor):
        if memory_format == torch.channels_last and data.dim() != 4:
            memory_format = torch.preserve_format
        if memory_format == torch.channels_last_3d and data.dim() != 5:
            memory_format = torch.preserve_format
        return data.to(device=device, dtype=dtype, memory_format=memory_format, non_blocking=True)
    if isinstance(data, Mapping):
        return type(data)(
            {key: to(value, device=device, dtype=dtype, memory_format=memory_format) for key, value in data.items()}
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return type(data)(to(value, device=device, dtype=dtype, memory_format=memory_format) for value in data)
    return data


def arch_invariant_rand(shape, dtype: torch.dtype, device, seed: int | None = None) -> torch.Tensor:
    array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(array).to(dtype=dtype, device=device)


def get_data_batch_size(data: Any) -> int:
    if isinstance(data, torch.Tensor):
        return len(data)
    if isinstance(data, Mapping):
        for value in data.values():
            try:
                return get_data_batch_size(value)
            except ValueError:
                pass
    if isinstance(data, Sequence) and data:
        return len(data) if isinstance(data[0], torch.Tensor) else get_data_batch_size(data[0])
    raise ValueError("unable to infer batch size")


def disabled_train(self, mode: bool = True):
    del mode
    return self


@contextmanager
def timer(name: str):
    started = time.perf_counter()
    yield
    from worldfoundry.core.distributed.logging import log

    log.info("{} takes {:.4f}s", name, time.perf_counter() - started)


__all__ = [
    "Color",
    "arch_invariant_rand",
    "disabled_train",
    "get_data_batch_size",
    "set_random_seed",
    "timer",
    "to",
]
