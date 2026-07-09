"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> misc -> step_tracker.py functionality."""

from multiprocessing import RLock

import torch
from jaxtyping import Int64
from torch import Tensor
from torch.multiprocessing import Manager


class StepTracker:
    """Step tracker implementation."""
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self):
        """Init."""
        self.lock = Manager().RLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()

    def set_step(self, step: int) -> None:
        """Set step.

        Args:
            step: The step.

        Returns:
            The return value.
        """
        with self.lock:
            self.step.fill_(step)

    def get_step(self) -> int:
        """Get step.

        Returns:
            The return value.
        """
        with self.lock:
            return self.step.item()
