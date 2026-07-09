"""LPIPS pairwise perceptual distance (torchmetrics backend)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.perceptual import resolve_device, to_tensor

NetType = Literal["alex", "vgg", "squeeze"]


def compute_lpips(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    net_type: NetType = "alex",
    device: str | None = None,
) -> float:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device_t = resolve_device(device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device_t)
    ref = to_tensor(reference, device_t)
    gen = to_tensor(generated, device_t)
    with __import__("torch").no_grad():
        return float(metric(ref, gen).item())


__all__ = ["NetType", "compute_lpips"]
