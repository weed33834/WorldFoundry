"""Time helpers shared by runtime and evaluation code."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import logging
import os
import time
from typing import Callable


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CudaSyncTimer:
    """Optional CUDA-synchronized timer usable as a context manager or decorator."""

    def __init__(
        self,
        name: str | None = None,
        *,
        flag_env: str = "SYNC_TIMER",
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self.flag_env = flag_env
        self.log_fn = log_fn
        self.elapsed_ms = 0.0
        self._enabled = False
        self._using_cuda = False

    def __enter__(self):
        self._enabled = os.environ.get(self.flag_env, "0") == "1"
        if not self._enabled:
            return None
        import torch

        self._using_cuda = torch.cuda.is_available()
        if self._using_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self._wall_start = time.perf_counter()
        return lambda: self.elapsed_ms

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if not self._enabled:
            return None
        if self._using_cuda:
            import torch

            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = float(self.start.elapsed_time(self.end))
        else:
            self.elapsed_ms = (time.perf_counter() - self._wall_start) * 1000.0
        if self.name is not None:
            message = f"{self.name} takes {self.elapsed_ms / 1000:.4f}s"
            if self.log_fn is not None:
                self.log_fn(message)
            else:
                logging.getLogger(__name__).info(message)
        return None

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


__all__ = ["CudaSyncTimer", "utc_now_iso"]
