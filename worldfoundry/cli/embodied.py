"""CLI entry points for embodied closed-loop evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .utils import json_dump


def _plan_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import (
        materialize_embodied_rollout_requests,
    )

    benchmarks = []
    total_requests = 0
    for benchmark in config.get("benchmarks") or ():
        requests = materialize_embodied_rollout_requests(benchmark)
        request_count = len(requests)
        total_requests += request_count
        benchmarks.append(
            {
                "id": benchmark.get("id"),
                "benchmark_id": benchmark.get("benchmark_id"),
                "request_count": request_count,
                "episodes_per_task": benchmark.get("episodes_per_task"),
                "max_tasks": benchmark.get("max_tasks"),
                "params": benchmark.get("params") or {},
            }
        )
    return {
        "id": config.get("id"),
        "output_dir": config.get("output_dir"),
        "model": config.get("model"),
        "server": config.get("server"),
        "docker": config.get("docker"),
        "benchmark_count": len(benchmarks),
        "request_count": total_requests,
        "benchmarks": benchmarks,
    }


def _handle_embodied_plan(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config

    config = load_canonical_embodied_config(args.config, output_dir=args.output_dir, server_url=args.server_url)
    payload = _plan_payload(config)
    if args.json:
        json_dump(payload)
    else:
        print(
            f"Embodied plan {payload['id']}: "
            f"benchmarks={payload['benchmark_count']}, requests={payload['request_count']}, "
            f"output_dir={payload['output_dir']}"
        )
        for benchmark in payload["benchmarks"]:
            print(
                f"  {benchmark['id']}: benchmark_id={benchmark['benchmark_id']}, "
                f"requests={benchmark['request_count']}"
            )
    return 0


def _handle_embodied_run(args: argparse.Namespace) -> int:
    if args.shard_id is not None and args.num_shards is None:
        print("error: --num-shards is required when --shard-id is set")
        return 2
    from worldfoundry.evaluation.tasks.embodied.orchestrator import run_embodied_eval_config

    result = run_embodied_eval_config(
        args.config,
        output_dir=args.output_dir,
        server_url=args.server_url,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        eval_id=args.eval_id,
        no_docker=args.no_docker,
        no_save=args.no_save,
        pull_docker=args.pull_docker,
    )
    payload = result.evaluate_result.to_dict()
    payload["eval_id"] = result.eval_id
    payload["raw_result_count"] = len(result.raw_results)
    if args.json:
        json_dump(payload)
    else:
        print(
            f"Embodied run {result.evaluate_result.status}: "
            f"samples={result.evaluate_result.sample_count}, "
            f"scorecard={result.evaluate_result.scorecard_path}"
        )
    return result.evaluate_result.exit_code


def _handle_embodied_merge(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
    from worldfoundry.evaluation.tasks.embodied.merge_results import merge_embodied_results

    config = load_canonical_embodied_config(args.config) if args.config else None
    metric_ids = tuple(args.metric or ("task_success", "success_rate"))
    result = merge_embodied_results(args.output_dir, eval_id=args.eval_id, metric_ids=metric_ids, config=config)
    payload = result.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(f"Embodied merge {result.status}: samples={result.sample_count}, scorecard={result.scorecard_path}")
    return result.exit_code


def _handle_embodied_serve(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.embodied.model_server.serve import serve_from_config

    serve_from_config(args.config, host=args.host, port=args.port)
    return 0


def register_embodied_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "embodied",
        help="Run embodied closed-loop model servers, evaluations, plans, and merges",
        description="Run embodied closed-loop model servers, evaluations, plans, and merges.",
    )
    embodied_subparsers = parser.add_subparsers(dest="embodied_command", required=True)

    plan_parser = embodied_subparsers.add_parser("plan", help="Materialize an embodied eval plan without running it")
    plan_parser.add_argument("--config", required=True, type=Path)
    plan_parser.add_argument("--output-dir", type=Path)
    plan_parser.add_argument("--server-url")
    plan_parser.add_argument("--json", action="store_true")
    plan_parser.set_defaults(func=_handle_embodied_plan)

    run_parser = embodied_subparsers.add_parser("run", help="Run an embodied eval config")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--server-url")
    run_parser.add_argument("--shard-id", type=int)
    run_parser.add_argument("--num-shards", type=int)
    run_parser.add_argument("--eval-id")
    run_parser.add_argument("--no-docker", action="store_true")
    run_parser.add_argument("--no-save", action="store_true")
    run_parser.add_argument("--pull-docker", action="store_true")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(func=_handle_embodied_run)

    merge_parser = embodied_subparsers.add_parser("merge", help="Merge embodied sharded rollout results")
    merge_parser.add_argument("--output-dir", required=True, type=Path)
    merge_parser.add_argument("--config", type=Path)
    merge_parser.add_argument("--eval-id")
    merge_parser.add_argument("--metric", action="append", default=None)
    merge_parser.add_argument("--json", action="store_true")
    merge_parser.set_defaults(func=_handle_embodied_merge)

    serve_parser = embodied_subparsers.add_parser("serve", help="Serve an embodied policy adapter over WebSocket")
    serve_parser.add_argument("--config", required=True, type=Path)
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.set_defaults(func=_handle_embodied_serve)


__all__ = ["register_embodied_subparser"]
