"""Realtime robotics frame display for Studio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RealtimeVisualizer:
    window_name: str = "WorldFoundry Realtime"
    wait_ms: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        self.running = bool(self.enabled)
        self._frame: Optional[np.ndarray] = None
        self._cv2 = None
        if self.running:
            import cv2

            self._cv2 = cv2
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)

    def update_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    def display(self) -> bool:
        if not self.running:
            return False
        if self._frame is not None:
            self._cv2.imshow(self.window_name, self._frame)
        key = self._cv2.waitKey(self.wait_ms) & 0xFF
        if key in (27, ord("q")):
            self.close()
        return self.running

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyWindow(self.window_name)
        self.running = False


__all__ = ["RealtimeVisualizer"]
