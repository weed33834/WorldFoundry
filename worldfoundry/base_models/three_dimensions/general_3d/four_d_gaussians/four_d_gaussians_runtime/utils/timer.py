"""Module for base_models -> three_dimensions -> general_3d -> four_d_gaussians -> four_d_gaussians_runtime -> utils -> timer.py functionality."""

import time
class Timer:
    """Timer implementation."""
    def __init__(self):
        """Init."""
        self.start_time = None
        self.elapsed = 0
        self.paused = False

    def start(self):
        """Start."""
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.start_time = time.time() - self.elapsed
            self.paused = False

    def pause(self):
        """Pause."""
        if not self.paused:
            self.elapsed = time.time() - self.start_time
            self.paused = True

    def get_elapsed_time(self):
        """Get elapsed time."""
        if self.paused:
            return self.elapsed
        else:
            return time.time() - self.start_time