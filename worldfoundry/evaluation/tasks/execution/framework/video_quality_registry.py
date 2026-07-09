"""Registry of per-benchmark video-quality contract configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping


@lru_cache(maxsize=None)
def get_video_quality_benchmark_config(benchmark_id: str) -> Mapping[str, Any]:
    key = benchmark_id.strip().lower()
    if key == "aigcbench":
        from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_video_quality_contract import (
            VIDEO_QUALITY_CONFIG,
        )

        return VIDEO_QUALITY_CONFIG
    if key == "mirabench":
        from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_video_quality_contract import (
            VIDEO_QUALITY_CONFIG,
        )

        return VIDEO_QUALITY_CONFIG
    if key == "fetv":
        from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_video_quality_contract import (
            VIDEO_QUALITY_CONFIG,
        )

        return VIDEO_QUALITY_CONFIG
    if key == "genai-bench":
        from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_video_quality_contract import (
            VIDEO_QUALITY_CONFIG,
        )

        return VIDEO_QUALITY_CONFIG
    if key == "ipv-bench":
        from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_video_quality_contract import (
            VIDEO_QUALITY_CONFIG,
        )

        return VIDEO_QUALITY_CONFIG
    known = ", ".join(supported_video_quality_benchmark_ids())
    raise KeyError(f"unsupported video-quality benchmark {benchmark_id!r}; known: {known}")


def supported_video_quality_benchmark_ids() -> tuple[str, ...]:
    return ("aigcbench", "fetv", "genai-bench", "ipv-bench", "mirabench")
