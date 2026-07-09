"""Module for base_models -> three_dimensions -> slam -> droid_slam -> cuda_timer.py functionality."""

import torch

class CudaTimer:
    """Cuda timer implementation."""
    def __init__(self, name, enabled=True):
        """Init.

        Args:
            name: The name.
            enabled: The enabled.
        """
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        """Enter."""
        if self.enabled:
            self.start.record()
        
    def __exit__(self, type, value, traceback):
        """Exit.

        Args:
            type: The type.
            value: The value.
            traceback: The traceback.
        """
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            print(self.name, elapsed)
