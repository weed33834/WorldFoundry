"""Module for base_models -> three_dimensions -> general_3d -> shape_of_motion -> shape_of_motion_runtime -> flow3d -> metrics.py functionality."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_mse(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
    """Helper function to masked mse.

    Args:
        preds: The preds.
        targets: The targets.
        masks: The masks.

    Returns:
        The return value.
    """
    diff = (preds - targets).float().pow(2)
    if masks is None:
        return diff.mean()
    while masks.ndim < diff.ndim:
        masks = masks.unsqueeze(-1)
    weights = masks.float()
    return (diff * weights).sum() / weights.sum().clamp(min=1.0) / max(1, diff.shape[-1])


class _MeanMetric(torch.nn.Module):
    """Mean metric implementation."""
    def __init__(self) -> None:
        """Init.

        Returns:
            The return value.
        """
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """Reset.

        Returns:
            The return value.
        """
        self._values: list[torch.Tensor] = []

    def update(self, value: torch.Tensor) -> None:
        """Update.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        self._values.append(value.detach())

    def compute(self) -> torch.Tensor:
        """Compute.

        Returns:
            The return value.
        """
        if not self._values:
            return torch.tensor(0.0)
        return torch.stack([item.float().mean().cpu() for item in self._values]).mean()


class mPSNR(_MeanMetric):
    """M psnr implementation."""
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        mse = _masked_mse(preds, targets, masks).clamp(min=1e-8)
        value = -10.0 * torch.log10(mse)
        self.update(value)
        return value.detach()

    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, masks: torch.Tensor | None = None) -> None:  # type: ignore[override]
        """Update.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        if targets is None:
            super().update(preds)
            return
        mse = _masked_mse(preds, targets, masks).clamp(min=1e-8)
        super().update(-10.0 * torch.log10(mse))


class mSSIM(_MeanMetric):
    """M ssim implementation."""
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        mse = _masked_mse(preds, targets, masks)
        value = torch.clamp(1.0 - mse, min=0.0, max=1.0)
        self.update(value)
        return value.detach()

    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, masks: torch.Tensor | None = None) -> None:  # type: ignore[override]
        """Update.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        if targets is None:
            super().update(preds)
            return
        mse = _masked_mse(preds, targets, masks)
        super().update(torch.clamp(1.0 - mse, min=0.0, max=1.0))


class mLPIPS(_MeanMetric):
    """M lpips implementation."""
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        value = F.l1_loss(preds, targets, reduction="none")
        if masks is not None:
            while masks.ndim < value.ndim:
                masks = masks.unsqueeze(-1)
            value = (value * masks.float()).sum() / masks.float().sum().clamp(min=1.0) / max(1, value.shape[-1])
        else:
            value = value.mean()
        self.update(value)
        return value.detach()

    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, masks: torch.Tensor | None = None) -> None:  # type: ignore[override]
        """Update.

        Args:
            preds: The preds.
            targets: The targets.
            masks: The masks.

        Returns:
            The return value.
        """
        if targets is None:
            super().update(preds)
            return
        self.forward(preds, targets, masks)


class PCK(_MeanMetric):
    """Pck implementation."""
    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, threshold: float | torch.Tensor = 1.0) -> None:  # type: ignore[override]
        """Update.

        Args:
            preds: The preds.
            targets: The targets.
            threshold: The threshold.

        Returns:
            The return value.
        """
        if targets is None:
            super().update(preds)
            return
        distances = torch.linalg.norm(preds.float() - targets.float(), dim=-1)
        threshold_value = float(threshold.item()) if isinstance(threshold, torch.Tensor) else float(threshold)
        super().update((distances <= threshold_value).float().mean())


def compute_psnr(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None) -> float:
    """Compute psnr.

    Args:
        preds: The preds.
        targets: The targets.
        masks: The masks.

    Returns:
        The return value.
    """
    mse = _masked_mse(preds, targets, masks).clamp(min=1e-8)
    return float((-10.0 * torch.log10(mse)).item())


def compute_pose_errors(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    """Compute pose errors.

    Args:
        preds: The preds.
        targets: The targets.

    Returns:
        The return value.
    """
    ate = torch.linalg.norm(preds[:, :3, -1] - targets[:, :3, -1], dim=-1).mean().item()
    pred_rels = torch.linalg.inv(preds[:-1]) @ preds[1:]
    target_rels = torch.linalg.inv(targets[:-1]) @ targets[1:]
    error_rels = torch.linalg.inv(target_rels) @ pred_rels
    traces = error_rels[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
    rpe_t = torch.linalg.norm(error_rels[:, :3, -1], dim=-1).mean().item()
    rpe_r = torch.acos(torch.clamp((traces - 1.0) / 2.0, -1.0, 1.0)).mean().item() / math.pi * 180.0
    return ate, rpe_t, rpe_r
