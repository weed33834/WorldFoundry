"""Shared inference-only LPIPS adapter."""

from __future__ import annotations

from typing import Any

import numpy as np


class LPIPSMetric:
    """Lazily load LPIPS and score NHWC uint8 image batches."""

    def __init__(self, *, net: str = "vgg", device: str = "auto") -> None:
        if net not in {"alex", "vgg", "vgg16", "squeeze"}:
            raise ValueError(f"unsupported LPIPS backbone: {net}")
        self.net = "vgg" if net == "vgg16" else net
        self.requested_device = device
        self.device: str | None = None
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError("LPIPSMetric requires the lpips and torch packages") from exc
        self.device = "cuda" if self.requested_device == "auto" and torch.cuda.is_available() else self.requested_device
        if self.device == "auto":
            self.device = "cpu"
        self._model = lpips.LPIPS(net=self.net).to(self.device).eval()
        return self._model

    def _prepare(self, images: np.ndarray) -> Any:
        import torch

        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"LPIPS input must be NHWC RGB, got {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"LPIPS input must be uint8, got {images.dtype}")
        return ((torch.from_numpy(images).to(self.device, dtype=torch.float32) / 127.5) - 1.0).permute(0, 3, 1, 2)

    def __call__(self, images_a: np.ndarray, images_b: np.ndarray, *, batch_size: int = 10) -> np.ndarray:
        import torch

        model = self._load()
        if len(images_a) != len(images_b):
            raise ValueError("LPIPS image batches must have equal length")
        values = []
        with torch.inference_mode():
            for start in range(0, len(images_a), batch_size):
                values.append(
                    model(
                        self._prepare(images_a[start : start + batch_size]),
                        self._prepare(images_b[start : start + batch_size]),
                    )
                    .reshape(-1)
                    .detach()
                    .cpu()
                )
        return torch.cat(values).numpy() if values else np.empty((0,), dtype=np.float32)
