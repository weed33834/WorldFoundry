"""Registry of per-benchmark in-tree evaluator configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping


@lru_cache(maxsize=None)
def get_in_tree_benchmark_config(benchmark_id: str) -> Mapping[str, object]:
    key = benchmark_id.strip().lower()
    if key == "t2v-safety-bench":
        from worldfoundry.evaluation.tasks.execution.runners.t2v_safety_bench.t2v_safety_bench_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    if key == "videoscience-bench":
        from worldfoundry.evaluation.tasks.execution.runners.videoscience_bench.videoscience_bench_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    if key == "phyeduvideo":
        from worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.phyeduvideo_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    if key == "worldarena":
        from worldfoundry.evaluation.tasks.execution.runners.worldarena.worldarena_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    if key == "world-in-world":
        from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    if key == "ewmbench":
        from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_in_tree_evaluator import IN_TREE_CONFIG
        return IN_TREE_CONFIG
    known = ", ".join(supported_in_tree_benchmark_ids())
    raise KeyError(f"unsupported in-tree benchmark {benchmark_id!r}; known: {known}")


def supported_in_tree_benchmark_ids() -> tuple[str, ...]:
    return (
        "ewmbench",
        "phyeduvideo",
        "t2v-safety-bench",
        "videoscience-bench",
        "world-in-world",
        "worldarena",
    )


def target_benchmark_metrics() -> dict[str, tuple[str, ...]]:
    return {
        benchmark_id: tuple(get_in_tree_benchmark_config(benchmark_id)["metric_ids"])  # type: ignore[arg-type]
        for benchmark_id in supported_in_tree_benchmark_ids()
    }


def __getattr__(name: str) -> object:
    if name == "TARGET_BENCHMARK_METRICS":
        return target_benchmark_metrics()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
