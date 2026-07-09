#!/usr/bin/env python3
"""Official runner for Physics-IQ image-to-video physical understanding evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_physics_iq_metrics,
    load_results_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
    CANONICAL_PROMPT_COUNT,
    load_description_rows,
    resolve_descriptions_path,
    resolve_physics_iq_root,
    unique_generation_records,
    video_stem_for_record,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_runtime import (
    run_physics_iq_scorer,
    scorer_config_from_env,
)

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize Physics-IQ official outputs.")
    parser.add_argument("--benchmark-id", default="physics-iq")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help=(
            "Evaluate WorldFoundry-generated Physics-IQ artifacts by importing a "
            "CSV/JSON results file from --generated-artifact-dir or "
            "WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--physics-iq-root", type=Path)
    parser.add_argument("--descriptions-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = direct_metrics.get(metric_id)
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": "physics_iq_official_runtime" if official_runtime_executed else "physics_iq_results_csv",
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_physics_iq_results",
            }
        )
    return rows


def _video_index(generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None:
        return {
            "provided": False,
            "generated_artifact_dir": None,
            "video_count": 0,
            "by_stem": {},
            "duplicate_stems": [],
        }
    by_stem: dict[str, list[str]] = {}
    if generated_dir.exists():
        for path in sorted(generated_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_stem.setdefault(path.stem, []).append(str(path))
    return {
        "provided": True,
        "generated_artifact_dir": str(generated_dir),
        "video_count": sum(len(paths) for paths in by_stem.values()),
        "by_stem": dict(by_stem),
        "duplicate_stems": sorted(stem for stem, paths in by_stem.items() if len(paths) > 1),
    }


def _coverage(expected_stems: set[str], generated_dir: Path | None) -> dict[str, Any]:
    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                actual_names.add(path.stem)
    missing = sorted(expected_stems - actual_names)
    unexpected = sorted(actual_names - expected_stems)
    matched = sorted(expected_stems & actual_names)
    return {
        "expected_count": len(expected_stems),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_stems) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    descriptions_path: Path | None,
    metric_rows: list[dict[str, Any]],
    video_coverage: Mapping[str, Any],
    scorer_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
    prompt_count: int,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("expected_count", 0) > 0
        and video_coverage.get("complete") is True
    )
    full_suite_valid = (
        prompt_count >= CANONICAL_PROMPT_COUNT
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    official_verified = official_runtime_executed and normalization_ok
    integration_evidence = official_verified and full_suite_valid
    normalizer_only = not official_runtime_executed
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": normalizer_only,
        "normalization_ok": normalization_ok,
        "eligibility": {
            "full_suite_valid": full_suite_valid,
            "video_coverage_complete": video_complete,
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_physics_iq_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "scorer_summary": dict(scorer_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "Physics-IQ"},
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "physics_iq_official_in_tree" if official_runtime_executed else "physics_iq_result_normalizer",
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "Physics-IQ in-tree runtime materializes descriptions.csv take-1 I2V prompts and normalizes official scorer CSV.",
            "Generation and scoring execution are handled by the unified WorldFoundry pipeline, not benchmark-local scripts.",
        ],
        "descriptions_path": None if descriptions_path is None else str(descriptions_path),
        "prompt_count": prompt_count,
    }


def normalize_physics_iq_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    scorer_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH, or --run-official is required"
        )
    repo_root = resolve_physics_iq_root(args.physics_iq_root)
    descriptions_path = resolve_descriptions_path(explicit=args.descriptions_file, repo_root=repo_root)
    prompt_records = unique_generation_records(load_description_rows(descriptions_path=descriptions_path))
    if args.limit is not None:
        prompt_records = prompt_records[: int(args.limit)]
    result_rows = load_results_rows(Path(official_results_path))
    computed = compute_physics_iq_metrics(
        rows=result_rows,
        description_rows=prompt_records,
    )
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        official_runtime_executed=official_runtime_executed,
    )
    expected_stems = {video_stem_for_record(record) for record in prompt_records}
    video_coverage = _coverage(expected_stems, generated_dir)
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        descriptions_path=descriptions_path,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        scorer_summary=scorer_summary,
        official_runtime_executed=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_PHYSICS_IQ_STRICT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    per_sample_rows = [
        {
            "scenario": row.get("scenario"),
            "category": row.get("category"),
            **{key: row.get(key) for key in row.keys() if key.startswith(("spatial_iou", "weighted_spatial", "v1_mse"))},
        }
        for row in result_rows
        if "scenario" in row
    ]
    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", per_sample_rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "descriptions_path": str(descriptions_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "prompt_count": len(prompt_records),
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_physics_iq(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")
    repo_root = resolve_physics_iq_root(args.physics_iq_root)
    descriptions_path = resolve_descriptions_path(explicit=args.descriptions_file, repo_root=repo_root)
    scorer_summary = run_physics_iq_scorer(
        generated_artifact_dir=Path(generated_dir),
        output_dir=output_dir,
        config=scorer_config_from_env(),
        descriptions_path=descriptions_path,
        limit=args.limit,
    )
    results_path = Path(str(scorer_summary.get("results_path") or scorer_summary.get("results_csv")))
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_path,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        physics_iq_root=args.physics_iq_root,
        descriptions_file=descriptions_path,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_physics_iq_results(
        normalize_args,
        official_runtime_executed=True,
        scorer_summary=scorer_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_official:
            scorecard = run_official_physics_iq(args)
        else:
            scorecard = normalize_physics_iq_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_physics_iq_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "Physics-IQ"},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "physics_iq_result_normalizer",
                "blocked_count": len(METRIC_ORDER),
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"physics-iq: failed ({exc})", file=sys.stderr)
        return 1

    payload = {
        "ok": scorecard.get("normalization_ok") is True,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        **scorecard,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"physics-iq: {scorecard.get('evaluation', {}).get('kind', 'unknown')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
