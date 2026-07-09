#!/usr/bin/env python3
"""WorldArena official-result normalizer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.time import utc_now_iso
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, write_json
from worldfoundry.base_models.capabilities import vbench_asset_path
from worldfoundry.evaluation.tasks.execution.framework.result_normalizer import OfficialResultsNormalizer
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_WORLDARENA_ROOT = RUNNER_ROOT / "runtime" / "video_quality"
DEFAULT_DIMENSIONS = (
    "action_following",
    "trajectory_accuracy",
    "semantic_alignment",
    "depth_accuracy",
    "aesthetic_quality",
    "background_consistency",
    "dynamic_degree",
    "flow_score",
    "photometric_smoothness",
    "motion_smoothness",
    "image_quality",
    "subject_consistency",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize WorldArena official video-quality outputs.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "worldarena"))
    parser.add_argument("--official-results-path", "--results-path", dest="official_results_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--generated-video-dir", "--generated-artifact-dir", dest="generated_video_dir", type=Path)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Run the in-tree WorldArena video-quality runtime and normalize its result artifact.",
    )
    parser.add_argument("--worldarena-root", type=Path, help="Override the in-tree WorldArena video_quality runtime root.")
    parser.add_argument("--config-path", type=Path, help="WorldArena config YAML. If omitted, one is generated.")
    parser.add_argument("--dimension", nargs="+", default=None, help="WorldArena metric dimensions to run.")
    parser.add_argument("--gt-data-dir", type=Path, help="Prepared WorldArena gt_dataset root.")
    parser.add_argument("--generated-dataset-dir", type=Path, help="Prepared WorldArena generated_dataset root.")
    parser.add_argument("--action-gt-data-dir", type=Path, help="Prepared action-following gt_dataset root.")
    parser.add_argument("--action-generated-dataset-dir", type=Path, help="Prepared action-following generated_dataset root.")
    parser.add_argument("--ckpt-root", type=Path, help="Root containing WorldArena metric checkpoints.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _runtime_root(args: argparse.Namespace) -> Path:
    explicit = args.worldarena_root or env_path("WORLDFOUNDRY_WORLDARENA_ROOT")
    return (explicit or DEFAULT_WORLDARENA_ROOT).expanduser().resolve()


def _ckpt_root(args: argparse.Namespace) -> Path:
    explicit = args.ckpt_root or env_path("WORLDFOUNDRY_WORLDARENA_CKPT_DIR") or env_path("WORLDFOUNDRY_CKPT_DIR")
    return (explicit or (REPO_ROOT / "cache" / "worldfoundry" / "checkpoints" / "worldarena")).expanduser().resolve()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _asset_path_or(asset_id: str, fallback: Path | str) -> str:
    try:
        return str(vbench_asset_path(asset_id))
    except (FileNotFoundError, KeyError):
        return str(fallback)


def _path_or_fallback(resolver, fallback: Path | str) -> str:
    try:
        return str(resolver())
    except (FileNotFoundError, KeyError, RuntimeError):
        return str(fallback)


def _generated_config(args: argparse.Namespace, runtime_root: Path, output_dir: Path) -> Path:
    from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba import checkpoint_path as vfimamba_checkpoint_path
    from worldfoundry.base_models.perception_core.optical_flow.raft import checkpoint_path as raft_checkpoint_path
    from worldfoundry.base_models.perception_core.optical_flow.sea_raft import (
        checkpoint_path as sea_raft_checkpoint_path,
        config_path as sea_raft_config_path,
    )

    run_root = output_dir / "worldarena_runtime"
    prepared_base = run_root / "prepared"
    generated_dataset = (
        args.generated_dataset_dir.expanduser().resolve()
        if args.generated_dataset_dir is not None
        else prepared_base / "generated_dataset"
    )
    gt_dataset = (
        args.gt_data_dir.expanduser().resolve()
        if args.gt_data_dir is not None
        else prepared_base / "gt_dataset"
    )
    action_generated = (
        args.action_generated_dataset_dir.expanduser().resolve()
        if args.action_generated_dataset_dir is not None
        else prepared_base / "generated_dataset_action_following"
    )
    action_gt = (
        args.action_gt_data_dir.expanduser().resolve()
        if args.action_gt_data_dir is not None
        else prepared_base / "gt_dataset_action_following"
    )
    ckpt_root = _ckpt_root(args)
    clip_b32 = _asset_path_or("vbench_clip_vit_b32_checkpoint", ckpt_root / "clip" / "ViT-B-32.pt")
    clip_l14 = _asset_path_or("vbench_clip_vit_l14_checkpoint", ckpt_root / "clip" / "ViT-L-14.pt")
    aesthetic_head = _asset_path_or(
        "vbench_aesthetic_linear_checkpoint",
        ckpt_root / "aesthetic" / "sa_0_4_vit_l_14_linear.pth",
    )
    dino_source = _asset_path_or("vbench_dino_source", ckpt_root / "dino")
    dino_weight = _asset_path_or("vbench_dino_vitb16_checkpoint", ckpt_root / "dino" / "dino_vitbase16_pretrain.pth")
    musiq_weight = _asset_path_or("vbench_musiq_spaq_checkpoint", ckpt_root / "pyiqa" / "musiq_spaq_ckpt-358bb6af.pth")
    raft_weight = _path_or_fallback(raft_checkpoint_path, ckpt_root / "raft" / "raft-things.pth")
    sea_raft_cfg = str(sea_raft_config_path("spring-M.json"))
    sea_raft_weight = _path_or_fallback(sea_raft_checkpoint_path, ckpt_root / "sea_raft" / "Tartan-C-T-TSKH-spring540x960-M.pth")
    vfimamba_weight = _path_or_fallback(vfimamba_checkpoint_path, ckpt_root / "VFIMamba.pkl")
    config = {
        "model_name": "worldarena",
        "data": {"gt_path": str(gt_dataset), "val_base": str(generated_dataset)},
        "data_action_following": {"gt_path": str(action_gt), "val_base": str(action_generated)},
        "save_path": str(run_root / "output"),
        "save_path_action_following": str(run_root / "output_action_following"),
        "ckpt": {
            "action_following": clip_b32,
            "semantic_alignment": {
                "caption": str(ckpt_root / "Qwen2.5-VL-7B-Instruct"),
                "CLIP": str(ckpt_root / "clip-vit-base-patch16"),
            },
            "depth_accuracy": str(ckpt_root / "Depth-Anything-V2-Small-hf"),
            "aesthetic_quality": {
                "clip": clip_l14,
                "aesthetic_head": aesthetic_head,
            },
            "background_consistency": {
                "clip": clip_b32,
                "raft": raft_weight,
            },
            "dynamic_degree": {"raft": raft_weight},
            "flow_score": {"raft": raft_weight},
            "photometric_smoothness": {
                "cfg": sea_raft_cfg,
                "model": sea_raft_weight,
            },
            "motion_smoothness": {"model": vfimamba_weight},
            "image_quality": {"musiq": musiq_weight},
            "subject_consistency": {
                "repo": dino_source,
                "weight": dino_weight,
                "model": "dino_vitb16",
                "raft": raft_weight,
            },
            "sam3_model_ckpt": str(ckpt_root / "sam3"),
            "vlm_model": str(ckpt_root / "Qwen3-VL-8B-Instruct"),
        },
    }
    config_path = run_root / "worldarena_config.yaml"
    _write_yaml(config_path, config)
    return config_path


def _latest_result_file(config_path: Path, dimensions: list[str]) -> Path:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    save_paths = [Path(cfg["save_path"])]
    action_path = cfg.get("save_path_action_following")
    if action_path:
        save_paths.append(Path(action_path))
    candidates: list[Path] = []
    for save_path in save_paths:
        if save_path.is_dir():
            candidates.extend(sorted(save_path.glob("*_results.json")))
    if not candidates:
        raise FileNotFoundError(f"WorldArena runtime did not write *_results.json under {save_paths}")
    if dimensions == ["action_following"]:
        action_candidates = [path for path in candidates if "action_following" in path.name]
        if action_candidates:
            return max(action_candidates, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_official_worldarena(args: argparse.Namespace, output_dir: Path) -> Path:
    runtime_root = _runtime_root(args)
    eval_script = runtime_root / "evaluate.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"missing in-tree WorldArena runtime: {eval_script}")
    config_path = args.config_path.expanduser().resolve() if args.config_path else _generated_config(args, runtime_root, output_dir)
    dimensions = args.dimension or list(DEFAULT_DIMENSIONS)
    command = [sys.executable, str(eval_script), "--dimension", *dimensions, "--config_path", str(config_path)]
    if args.overwrite:
        command.append("--overwrite")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(command, cwd=str(runtime_root), env=env, text=True, capture_output=True, check=False)
    log_path = output_dir / "worldarena_official_runtime.log"
    log_path.write_text((proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"WorldArena official runtime failed with code {proc.returncode}; see {log_path}")
    return _latest_result_file(config_path, list(dimensions))


def resolve_results_path(args: argparse.Namespace) -> Path | None:
    if args.official_results_path is not None:
        return args.official_results_path.expanduser().resolve()
    env_result = env_path("WORLDFOUNDRY_WORLDARENA_RESULTS_PATH")
    if env_result is not None:
        return env_result.expanduser().resolve()
    generated_dir = args.generated_video_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None:
        return None
    root = generated_dir.expanduser().resolve()
    for candidate in (
        root / "worldarena_results.json",
        root / "results.json",
        root / "summary.json",
    ):
        if candidate.is_file():
            return candidate
    matches = sorted(root.glob("*_results.json"))
    return matches[-1].resolve() if matches else None


def normalize_worldarena_results(args: argparse.Namespace, results_path: Path, output_dir: Path) -> dict[str, Any]:
    entry = load_benchmark_zoo_registry(BENCHMARK_ZOO_DIR).get(args.benchmark_id)
    normalization = OfficialResultsNormalizer.from_benchmark_entry(entry).normalize_file(str(results_path))
    per_metric = normalization.scorecard_metrics()
    available_count = sum(1 for item in per_metric.values() if item.get("available") is True)
    raw_metric_path = output_dir / "raw_metric_table.jsonl"
    raw_metric_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalization.raw_metric_rows()),
        encoding="utf-8",
    )
    scorecard_path = output_dir / "scorecard.json"
    official_runtime_executed = bool(getattr(args, "run_official", False))
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_runtime_executed and available_count > 0,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": available_count > 0,
        "run": {
            "status": "official_results_imported" if available_count > 0 else "official_results_missing_scores",
            "started_at": utc_now_iso(),
            "runner": "worldarena_official_runner",
        },
        "benchmark": {
            "benchmark_id": args.benchmark_id,
            "contract_only": False,
            "evidence_level": "official_results_normalized",
        },
        "dataset": {
            "official_results_path": str(results_path.resolve()),
            "generated_video_dir": None if args.generated_video_dir is None else str(args.generated_video_dir),
        },
        "metrics": {
            "leaderboard": {
                metric_id: item["raw_score"]
                for metric_id, item in per_metric.items()
                if item.get("available") and item.get("raw_score") is not None
            },
            "per_metric": per_metric,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_path.resolve()),
            "official_results": str(results_path.resolve()),
        },
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_official:
        try:
            results_path = run_official_worldarena(args, output_dir)
        except Exception as exc:  # noqa: BLE001
            failure = {
                "schema_version": SCORECARD_SCHEMA_VERSION,
                "official_benchmark_verified": False,
                "integration_evidence": False,
                "leaderboard_valid": False,
                "normalization_ok": False,
                "run": {
                    "status": "failed",
                    "started_at": utc_now_iso(),
                    "runner": "worldarena_official_runner",
                    "error": str(exc),
                },
                "benchmark": {"benchmark_id": args.benchmark_id},
                "metrics": {"leaderboard": {}, "per_metric": {}},
                "artifacts": {"scorecard": str((output_dir / "scorecard.json").resolve())},
            }
            write_json(output_dir / "scorecard.json", failure)
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc), "scorecard": failure}, indent=2, ensure_ascii=False))
            else:
                print(f"{args.benchmark_id}: failed: {exc}", file=sys.stderr)
            return 1
    else:
        results_path = resolve_results_path(args)
    if results_path is None or not results_path.exists():
        print("error: --official-results-path or WORLDFOUNDRY_WORLDARENA_RESULTS_PATH is required", file=sys.stderr)
        return 2
    scorecard = normalize_worldarena_results(args, results_path, output_dir)
    available_count = sum(1 for item in scorecard["metrics"]["per_metric"].values() if item.get("available") is True)
    payload = {
        "ok": available_count > 0,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(output_dir),
        "normalization_ok": available_count > 0,
        "scorecard": scorecard,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"{args.benchmark_id}: normalized {available_count} metrics")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
