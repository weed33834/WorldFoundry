"""Minimal helpers used by the in-tree HY-World trajectory renderer."""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank0_log(message: object, level: str = "INFO") -> None:
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[{level.upper()}] {message}", flush=True)


class Timer:
    def __init__(self) -> None:
        self.records: defaultdict[str, list[float]] = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.records[name].append(time.perf_counter() - started)

    def summary(self) -> None:
        for name, values in self.records.items():
            total = sum(values)
            rank0_log(f"{name}: {total:.3f}s total across {len(values)} call(s)")


def split_n_into_d_parts(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


__all__ = ["Timer", "rank0_log", "set_seed", "split_n_into_d_parts"]
