#!/usr/bin/env python3
"""Official runner for T2VWorldBench.

Normalize ``*_video_assessment_scores.csv`` or run ``eval.py`` on a video directory.

Environment variables
---------------------
- ``WORLDFOUNDRY_T2VWORLDBENCH_ROOT``
- ``WORLDFOUNDRY_T2VWORLDBENCH_RESULTS_PATH``
- ``WORLDFOUNDRY_T2VWORLDBENCH_PROMPT_FILE``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, mean_numeric, normalize_unit_score, scalar_number



DEFAULT_PROMPT_FILE = bundled_benchmark_asset("t2vworldbench", "data", "meta_data", "meta_data.json")

CONFIG = ors.BenchRunnerConfig(
    benchmark_id='t2vworldbench',
    display_name='T2VWorldBench',
    root_env='WORLDFOUNDRY_T2VWORLDBENCH_ROOT',
    results_path_env='WORLDFOUNDRY_T2VWORLDBENCH_RESULTS_PATH',
    default_repo_subdir='worldfoundry/evaluation/tasks/execution/runners/t2vworldbench/runtime/t2vworldbench',
    metric_order=('physics_knowledge', 'nature_knowledge', 'activity_knowledge', 'culture_knowledge', 'causality_knowledge', 'object_knowledge', 'world_knowledge_average'),
    metric_specs={'physics_knowledge': {'name': 'Physics Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'nature_knowledge': {'name': 'Nature Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'activity_knowledge': {'name': 'Activity Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'culture_knowledge': {'name': 'Culture Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'causality_knowledge': {'name': 'Causality Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'object_knowledge': {'name': 'Object Knowledge', 'group': 'world_knowledge', 'higher_is_better': True}, 'world_knowledge_average': {'name': 'World Knowledge Average', 'group': 'aggregate', 'higher_is_better': True}},
    metric_aliases={'physics': 'physics_knowledge', 'physics_knowledge': 'physics_knowledge', 'nature': 'nature_knowledge', 'nature_knowledge': 'nature_knowledge', 'activity': 'activity_knowledge', 'activity_knowledge': 'activity_knowledge', 'culture': 'culture_knowledge', 'culture_knowledge': 'culture_knowledge', 'causality': 'causality_knowledge', 'causality_knowledge': 'causality_knowledge', 'object': 'object_knowledge', 'object_knowledge': 'object_knowledge', 'avg': 'world_knowledge_average', 'average': 'world_knowledge_average', 'world_knowledge_average': 'world_knowledge_average'},
    average_metric_id='world_knowledge_average',
    official_entry='eval.py',
    official_output_globs=('results/*_video_assessment_scores.csv', '*_video_assessment_scores.csv'),
    requires_api_env=(),
    usage_epilog='Examples:\n  python3 run_t2vworldbench_official_runner.py \\\n    --official-results-path my_model_video_assessment_scores.csv \\\n    --output-dir /tmp/t2vworld_out --json\n\n  python3 run_t2vworldbench_official_runner.py --run-official \\\n    --model-name my_model --prompt-file prompts.txt \\\n    --generated-video-dir /path/to/videos --output-dir /tmp/t2vworld_out --json',
)

CATEGORY_COLUMNS = {
    "physics": "physics_knowledge",
    "nature": "nature_knowledge",
    "activity": "activity_knowledge",
    "culture": "culture_knowledge",
    "causality": "causality_knowledge",
    "object": "object_knowledge",
}

def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir, output_dir / "upstream"]
    if repo_root is not None:
        search_roots.extend([repo_root, repo_root / "results"])
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)

def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    extracted = ors.generic_extract_metrics(payload, CONFIG, str(results_path))
    if extracted or not isinstance(payload, list):
        return extracted
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in CATEGORY_COLUMNS.values()}
    for row in payload:
        if not isinstance(row, dict):
            continue
        for source_col, metric_id in CATEGORY_COLUMNS.items():
            score = scalar_number(row.get(source_col) if row.get(source_col) is not None else row.get(source_col.title()))
            if score is not None:
                buckets[metric_id].append(score)
    for metric_id, values in buckets.items():
        avg = mean_numeric(values)
        if avg is None:
            continue
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": avg,
            "normalized_score": normalize_unit_score(avg),
            "source": "t2vworldbench_assessment_csv",
            "sample_count": len(values),
        }
    return extracted

def build_official_command(*, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any) -> list[str] | None:
    prompt_file = args.prompt_file or env_path("WORLDFOUNDRY_T2VWORLDBENCH_PROMPT_FILE") or DEFAULT_PROMPT_FILE
    if prompt_file is None:
        return None
    model_name = args.model_name or os.environ.get("WORLDFOUNDRY_T2VWORLDBENCH_MODEL_NAME", "model")
    upstream_output = output_dir / "upstream"
    upstream_output.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(repo_root / config.official_entry),
        "--video-path",
        str(generated_video_dir),
        "--t2v-model",
        model_name,
        "--read-prompt-file",
        str(prompt_file),
        "--output-path",
        str(upstream_output),
    ]

def extend_parser(parser) -> None:
    parser.add_argument("--model-name", default=None, help="upstream --t2v-model name")
    parser.add_argument("--prompt-file", type=Path, help="prompt file passed to eval.py")

def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
