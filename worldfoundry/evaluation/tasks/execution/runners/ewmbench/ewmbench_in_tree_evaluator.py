"""In-tree evaluator contract for ewmbench."""

from __future__ import annotations

from collections.abc import Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
    evaluate_benchmark_metrics as _evaluate_benchmark_metrics,
)

BENCHMARK_ID = 'ewmbench'

IN_TREE_CONFIG: dict[str, object] = {
    "benchmark_id": BENCHMARK_ID,
    "evaluator_kind": 'reasoning',
    "metric_ids": ('scene_consistency', 'motion_correctness', 'semantic_alignment', 'diversity', 'ewmbench_average'),
    "primary_metric": 'ewmbench_average',
    "required_artifacts": ('generated_video',),}


class EwmbenchInTreeEvaluator(BenchmarkZooInTreeEvaluator):
    """Focused in-tree evaluator for ewmbench."""

    def __init__(
        self,
        *,
        metric_ids: Sequence[str] | None = None,
        required_artifacts: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            BENCHMARK_ID,
            metric_ids=metric_ids,
            required_artifacts=required_artifacts,
        )


def evaluate_ewmbench_metrics(
    request: GenerationRequest,
    result: GenerationResult,
    *,
    metric_ids: Sequence[str] | None = None,
    required_artifacts: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    return _evaluate_benchmark_metrics(
        BENCHMARK_ID,
        request,
        result,
        metric_ids=metric_ids,
        required_artifacts=required_artifacts,
    )
