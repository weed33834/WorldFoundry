"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> misc -> benchmarker.py functionality."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    """Benchmarker implementation."""
    def __init__(self) -> None:
        """Init.

        Returns:
            The return value.
        """
        self.execution_times = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        """Time.

        Args:
            tag: The tag.
            num_calls: The num calls.
        """
        try:
            start_time = time()
            yield
        finally:
            elapsed = time() - start_time
            for _ in range(num_calls):
                self.execution_times[tag].append(elapsed / num_calls)

    def dump(self, path: str | Path) -> None:
        """Dump.

        Args:
            path: The path.

        Returns:
            The return value.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: str | Path) -> None:
        """Dump memory.

        Args:
            path: The path.

        Returns:
            The return value.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        peak = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)
        with target.open("w") as f:
            json.dump(peak, f)

    def summarize(self) -> None:
        """Summarize.

        Returns:
            The return value.
        """
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
