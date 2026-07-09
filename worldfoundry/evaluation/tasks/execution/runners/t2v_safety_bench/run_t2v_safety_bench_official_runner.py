#!/usr/bin/env python3
"""Official runner for T2VSafetyBench.

Normalize ``nsfw_results_*.xlsx`` or run ``main.py`` with prompt file and GPT API key.

Environment variables
---------------------
- ``WORLDFOUNDRY_T2V_SAFETY_BENCH_ROOT``
- ``WORLDFOUNDRY_T2V_SAFETY_BENCH_RESULTS_PATH``
- ``WORLDFOUNDRY_T2V_SAFETY_BENCH_PROMPT_PATH``
- ``OPENAI_API_KEY`` / ``WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_API_KEY``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, mean_numeric, normalize_unit_score, scalar_number

import re as _re

CONFIG = ors.BenchRunnerConfig(
    benchmark_id='t2v-safety-bench',
    display_name='T2VSafetyBench',
    root_env='WORLDFOUNDRY_T2V_SAFETY_BENCH_ROOT',
    results_path_env='WORLDFOUNDRY_T2V_SAFETY_BENCH_RESULTS_PATH',
    default_repo_subdir='worldfoundry/evaluation/tasks/execution/runners/t2v_safety_bench/runtime/t2v_safety_bench',
    metric_order=('pornography_nsfw_rate', 'borderline_pornography_nsfw_rate', 'violence_nsfw_rate', 'gore_nsfw_rate', 'public_figures_nsfw_rate', 'discrimination_nsfw_rate', 'political_sensitivity_nsfw_rate', 'illegal_activities_nsfw_rate', 'disturbing_content_nsfw_rate', 'misinformation_falsehoods_nsfw_rate', 'copyright_trademark_nsfw_rate', 'temporal_risk_nsfw_rate', 'nsfw_average'),
    metric_specs={'pornography_nsfw_rate': {'name': 'Pornography Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'borderline_pornography_nsfw_rate': {'name': 'Borderline Pornography Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'violence_nsfw_rate': {'name': 'Violence Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'gore_nsfw_rate': {'name': 'Gore Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'public_figures_nsfw_rate': {'name': 'Public Figures Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'discrimination_nsfw_rate': {'name': 'Discrimination Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'political_sensitivity_nsfw_rate': {'name': 'Political Sensitivity Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'illegal_activities_nsfw_rate': {'name': 'Illegal Activities Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'disturbing_content_nsfw_rate': {'name': 'Disturbing Content Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'misinformation_falsehoods_nsfw_rate': {'name': 'Misinformation Falsehoods Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'copyright_trademark_nsfw_rate': {'name': 'Copyright Trademark Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'temporal_risk_nsfw_rate': {'name': 'Temporal Risk Nsfw Rate', 'group': 'safety', 'higher_is_better': True}, 'nsfw_average': {'name': 'Nsfw Average', 'group': 'safety', 'higher_is_better': True}},
    metric_aliases={'nsfw_generation_rate': 'nsfw_average', 'nsfw_average': 'nsfw_average', 'temporal_risk': 'temporal_risk_nsfw_rate'},
    average_metric_id='nsfw_average',
    official_entry='main.py',
    official_output_globs=('nsfw_results_*.xlsx', 'nsfw_results_*.txt'),
    requires_api_env=('WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_API_KEY', 'OPENAI_API_KEY'),
    usage_epilog='Examples:\n  python3 run_t2v_safety_bench_official_runner.py \\\n    --official-results-path nsfw_results_model_class0.xlsx \\\n    --output-dir /tmp/t2v_safety_out --json\n\n  export OPENAI_API_KEY=...\n  python3 run_t2v_safety_bench_official_runner.py --run-official \\\n    --model-name my_model --prompt-path prompts.json \\\n    --output-dir /tmp/t2v_safety_out --json',
)
T2V_SAFETY_ASSETS_ROOT = bundled_benchmark_asset("t2v-safety-bench")

def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir]
    if repo_root is not None:
        search_roots.append(repo_root)
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)

def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    extracted = ors.generic_extract_metrics(payload, CONFIG, str(results_path))
    if extracted:
        return extracted
    if results_path.suffix.lower() == ".txt":
        text = results_path.read_text(encoding="utf-8")
        match = _re.search(r"NSFW generation rate:\s*([0-9.]+)", text)
        if match:
            rate = float(match.group(1))
            extracted["nsfw_average"] = {
                "metric_id": "nsfw_average",
                "raw_score": rate,
                "normalized_score": normalize_unit_score(rate),
                "source": "t2v_safety_txt_summary",
                "sample_count": None,
            }
    return extracted

def build_official_command(*, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any) -> list[str] | None:
    api_key = ors.first_env_value(*config.requires_api_env)
    prompt_path = args.prompt_path or env_path("WORLDFOUNDRY_T2V_SAFETY_BENCH_PROMPT_PATH")
    if prompt_path is None:
        prompt_path = T2V_SAFETY_ASSETS_ROOT / "T2VSafetyBench" / f"{args.classes}.txt"
    if not api_key or prompt_path is None:
        return None
    if not Path(prompt_path).is_file():
        return None
    model_name = args.model_name or os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_MODEL_NAME", "model")
    save_dir = output_dir / "upstream"
    save_dir.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(repo_root / config.official_entry),
        "--video-model",
        model_name,
        "--prompt-path",
        str(prompt_path),
        "--classes",
        str(args.classes),
        "--save-dir",
        str(save_dir),
        "--gpt-api",
        api_key,
    ]

def extend_parser(parser) -> None:
    parser.add_argument("--model-name", default=None, help="upstream --video-model name")
    parser.add_argument("--classes", type=int, default=int(os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS", "1")))
    parser.add_argument("--prompt-path", type=Path, help="prompt JSON for main.py")

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
