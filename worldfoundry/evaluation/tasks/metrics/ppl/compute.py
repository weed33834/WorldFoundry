"""Perceptual Path Length computation via in-tree torch-fidelity."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics


def _parse_numeric_result(result: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_ppl(
    generative_model: Any,
    *,
    batch_size: int = 64,
    cuda: bool = True,
    num_samples: int = 5000,
    **kwargs: Any,
) -> dict[str, float]:
    result = calculate_metrics()(
        input1=generative_model,
        batch_size=batch_size,
        cuda=cuda,
        ppl=True,
        input1_model_num_samples=num_samples,
        **kwargs,
    )
    return _parse_numeric_result(result)


__all__ = ["compute_ppl"]
