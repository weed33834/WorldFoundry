"""Model-agnostic cross-step caches for diffusion inference.

The fixed-step and accumulated-change policies are adapted from generic
StepCache and EasyCache/TeaCache policies in NVIDIA Sol-Engine/SGLang. See the
third-party notices in ``worldfoundry/core/kernels``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


def _detach(value: T) -> T:
    if isinstance(value, torch.Tensor):
        return value.detach()  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)  # type: ignore[return-value]
    if isinstance(value, list):
        return [_detach(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}  # type: ignore[return-value]
    return value


def _tree_sub(left: T, right: T) -> T:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left - right  # type: ignore[return-value]
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_tree_sub(a, b) for a, b in zip(left, right, strict=True))  # type: ignore[return-value]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [_tree_sub(a, b) for a, b in zip(left, right, strict=True)]  # type: ignore[return-value]
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return {key: _tree_sub(left[key], right[key]) for key in left}  # type: ignore[return-value]
    raise TypeError("cached output trees must have matching tensor containers")


def _tree_add_scaled(value: T, delta: T, scale: float) -> T:
    if isinstance(value, torch.Tensor) and isinstance(delta, torch.Tensor):
        return value + delta * scale  # type: ignore[return-value]
    if isinstance(value, tuple) and isinstance(delta, tuple) and len(value) == len(delta):
        return tuple(_tree_add_scaled(a, b, scale) for a, b in zip(value, delta, strict=True))  # type: ignore[return-value]
    if isinstance(value, list) and isinstance(delta, list) and len(value) == len(delta):
        return [_tree_add_scaled(a, b, scale) for a, b in zip(value, delta, strict=True)]  # type: ignore[return-value]
    if isinstance(value, dict) and isinstance(delta, dict) and value.keys() == delta.keys():
        return {key: _tree_add_scaled(value[key], delta[key], scale) for key in value}  # type: ignore[return-value]
    raise TypeError("cached output trees must have matching tensor containers")


def _contains_grad_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, (tuple, list)):
        return any(_contains_grad_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_grad_tensor(item) for item in value.values())
    return False


@dataclass(frozen=True, slots=True)
class CacheEvent:
    """One cache decision, suitable for runtime telemetry."""

    step: int
    hit: bool
    reason: str
    accumulated_change: float | None = None


class FixedStepCache(Generic[T]):
    """Reuse a previous denoiser output on an explicit set of steps.

    ``delta_scale=0`` replays the last dense output. A non-zero value performs
    first-order extrapolation from the two latest dense outputs.
    """

    def __init__(
        self,
        skip_steps: Iterable[int] = (),
        *,
        delta_scale: float = 0.0,
        dense_first: int = 1,
        dense_last: int = 1,
        total_steps: int | None = None,
    ) -> None:
        self.skip_steps = frozenset(int(step) for step in skip_steps)
        if any(step < 0 for step in self.skip_steps):
            raise ValueError("skip_steps must be non-negative")
        if dense_first < 0 or dense_last < 0:
            raise ValueError("dense boundaries must be non-negative")
        if total_steps is not None and total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.delta_scale = float(delta_scale)
        self.dense_first = int(dense_first)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self._last: T | None = None
        self._delta: T | None = None
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        self._last = None
        self._delta = None
        self.events.clear()

    def _is_boundary(self, step: int, total_steps: int | None) -> bool:
        if step < self.dense_first:
            return True
        count = self.total_steps if total_steps is None else total_steps
        return count is not None and step >= max(count - self.dense_last, 0)

    def run(self, step: int, compute: Callable[[], T], *, total_steps: int | None = None) -> T:
        """Compute or replay one step output."""

        step = int(step)
        skip = step in self.skip_steps
        reason = "scheduled"
        if self._last is None:
            skip, reason = False, "seed"
        elif self._is_boundary(step, total_steps):
            skip, reason = False, "dense-boundary"
        elif torch.is_grad_enabled():
            skip, reason = False, "autograd"

        if skip:
            self.events.append(CacheEvent(step=step, hit=True, reason=reason))
            if self.delta_scale and self._delta is not None:
                return _tree_add_scaled(self._last, self._delta, self.delta_scale)
            return self._last

        output = compute()
        previous = self._last
        detached = _detach(output)
        if previous is not None:
            self._delta = _detach(_tree_sub(detached, previous))
        self._last = detached
        dense_reason = reason if step in self.skip_steps else "dense"
        self.events.append(CacheEvent(step=step, hit=False, reason=dense_reason))
        return output


class AdaptiveResidualCache:
    """Reuse a model residual while accumulated input change stays small.

    Warm up densely, estimate normalized change from a cheap signal, accumulate
    it across steps, and replay the last dense residual until the threshold or
    consecutive-hit cap is reached.
    """

    def __init__(
        self,
        threshold: float,
        *,
        warmup_steps: int = 1,
        max_consecutive_hits: int = 3,
        dense_last: int = 1,
        total_steps: int | None = None,
        subsample: int = 1,
        eps: float = 1e-6,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if warmup_steps < 0 or max_consecutive_hits < 0 or dense_last < 0:
            raise ValueError("cache step limits must be non-negative")
        if subsample < 1:
            raise ValueError("subsample must be positive")
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.max_consecutive_hits = int(max_consecutive_hits)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self.subsample = int(subsample)
        self.eps = float(eps)
        self._previous_signal: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        self._previous_signal = None
        self._residual = None
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        self.events.clear()

    def _sample(self, signal: torch.Tensor) -> torch.Tensor:
        detached = signal.detach()
        if self.subsample == 1 or detached.ndim < 2:
            return detached
        slices = [slice(None)] * detached.ndim
        for dim in range(1, detached.ndim - 1):
            slices[dim] = slice(None, None, self.subsample)
        return detached[tuple(slices)]

    def _relative_change(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        current_fp32 = current.float()
        previous_fp32 = previous.to(device=current.device).float()
        numerator = (current_fp32 - previous_fp32).abs().mean()
        denominator = previous_fp32.abs().mean().clamp_min(self.eps)
        return float((numerator / denominator).item())

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Return a dense or cached residual for one denoising step."""

        if torch.is_grad_enabled():
            residual = compute_residual()
            self.events.append(CacheEvent(int(step), False, "autograd"))
            return residual

        sampled = self._sample(signal)
        count = self.total_steps if total_steps is None else total_steps
        dense_boundary = int(step) < self.warmup_steps or (
            count is not None and int(step) >= max(int(count) - self.dense_last, 0)
        )

        relative_change = None
        if self._previous_signal is not None:
            relative_change = self._relative_change(sampled, self._previous_signal)
            self._accumulated_change += relative_change

        can_hit = (
            not dense_boundary
            and self._residual is not None
            and self._previous_signal is not None
            and self._accumulated_change < self.threshold
            and self._consecutive_hits < self.max_consecutive_hits
        )
        self._previous_signal = sampled.clone()
        if can_hit:
            self._consecutive_hits += 1
            self.events.append(CacheEvent(int(step), True, "below-threshold", self._accumulated_change))
            return self._residual

        residual = compute_residual()
        self._residual = residual.detach()
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        reason = "dense-boundary" if dense_boundary else "threshold"
        if relative_change is None:
            reason = "seed"
        self.events.append(CacheEvent(int(step), False, reason, relative_change))
        return residual


__all__ = ["AdaptiveResidualCache", "CacheEvent", "FixedStepCache"]
