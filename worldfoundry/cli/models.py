"""CLI commands for inspecting model runners and managing base-model assets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.utils import write_json

from .utils import json_dump

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT

# ── Models list and runtime runners ─────────────────────────────


def _handle_models_list(args: argparse.Namespace) -> int:
    """List available model types from the model registry."""
    from worldfoundry.evaluation.models.catalog.registry import discover_model_registry

    registry = discover_model_registry()
    items = [item.to_dict() for item in registry.list(args.family)]
    if args.json:
        json_dump(items)
        return 0

    for item in items:
        print(f"{item['model_type']} [{item['family']}] loader={item['has_loader']} infer={item['has_infer']}")
    return 0


def _handle_models_runtime_runners(args: argparse.Namespace) -> int:
    """Emit registered ``module:Class`` runner targets usable with ``worldfoundry-eval evaluate --mode model``.

    Parameters:
        args: CLI namespace; ``json`` selects JSON lines instead of plain text rows.
    """
    from worldfoundry.evaluation.models.runners.registry import model_runner_registry_report

    report = model_runner_registry_report()
    payload = [entry.to_dict() for entry in report.entries]
    if args.json:
        json_dump(payload)
        return 0
    for entry in payload:
        aliases = ", ".join(entry["aliases"]) if entry["aliases"] else "-"
        print(f"{entry['name']}: {entry['runner_target']} source={entry['source']} aliases={aliases}")
    for issue in report.issues:
        print(f"warning:{issue.code} {issue.name or '-'}: {issue.message}", file=sys.stderr)
    return 0


# ── Base-model asset management ──────────────────────────────────


def _execute_download_commands(commands: list[list[str]]) -> list[dict[str, object]]:
    """Run a sequence of download commands and capture truncated stdout/stderr."""
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    executed = []
    for command in commands:
        completed = subprocess.run(
            [str(item) for item in command],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        executed.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    return executed


def _handle_models_assets(args: argparse.Namespace) -> int:
    """Plan or download reusable base-model checkpoints and data assets."""
    from worldfoundry.base_models.capabilities import base_model_inventory, base_model_materialization_plan

    if args.list:
        inventory = base_model_inventory()
        if args.report_path is not None:
            write_json(args.report_path, inventory)
        if args.json:
            json_dump(inventory)
        else:
            print(f"base-model capabilities: {inventory['capability_count']}")
            for item in inventory["capabilities"]:
                print(f"capability {item['id']} [{item['family']}]")
            print(f"base-model stacks: {inventory['stack_count']}")
            for item in inventory["stacks"]:
                print(f"stack {item['id']} [{item['family']}] -> {', '.join(item['capability_ids'])}")
            if args.report_path is not None:
                print("report:", args.report_path)
        return 0

    plan = base_model_materialization_plan(args.capability)
    executed = []
    if args.execute_downloads:
        executed = _execute_download_commands(plan.get("download_command_argvs", []))
        plan = base_model_materialization_plan(args.capability)
        plan["executed_downloads"] = executed

    if args.report_path is not None:
        write_json(args.report_path, plan)

    if args.json:
        json_dump(plan)
    else:
        print("base-model assets:", "ok" if plan["ok"] else "missing")
        if plan["stack_ids"]:
            print("stacks:", ", ".join(plan["stack_ids"]))
        print("capabilities:", ", ".join(plan["capability_ids"]))
        if plan["pip_install_packages"]:
            print("install:", "python -m pip install " + " ".join(plan["pip_install_packages"]))
        for command in plan["download_commands"]:
            print("download:", command)
        for command in plan["export_commands"]:
            print("env:", command)
        for action in plan["manual_actions"]:
            print("manual:", action)
        if args.report_path is not None:
            print("report:", args.report_path)
    if not args.execute_downloads:
        return 0
    execution_failed = any(int(item.get("returncode", 1)) != 0 for item in executed)
    return 0 if plan["ok"] and not execution_failed else 1


def register_model_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES, BASE_MODEL_STACKS

    models_parser = subparsers.add_parser("models", help="Inspect registered model runners")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)

    models_list_parser = models_subparsers.add_parser("list", help="List available model types")
    models_list_parser.add_argument("--family")
    models_list_parser.add_argument("--json", action="store_true")
    models_list_parser.set_defaults(func=_handle_models_list)

    models_runtime_runners_parser = models_subparsers.add_parser(
        "runtime-runners", help="List registered runner targets usable with evaluate --mode model"
    )
    models_runtime_runners_parser.add_argument("--json", action="store_true")
    models_runtime_runners_parser.set_defaults(func=_handle_models_runtime_runners)

    models_assets_parser = models_subparsers.add_parser(
        "assets",
        help="Plan or download reusable base-model assets",
        description="Plan or download reusable base-model assets such as depth, SLAM, detection, segmentation, and motion stacks.",
    )
    models_assets_parser.add_argument(
        "--capability",
        action="append",
        choices=sorted([*BASE_MODEL_CAPABILITIES, *BASE_MODEL_STACKS]),
        help="Capability or stack id to materialize. May repeat. Defaults to all registered capabilities.",
    )
    models_assets_parser.add_argument(
        "--list", action="store_true", help="List registered base-model capabilities and stacks."
    )
    models_assets_parser.add_argument("--execute-downloads", action="store_true")
    models_assets_parser.add_argument("--report-path", type=Path)
    models_assets_parser.add_argument("--json", action="store_true")
    models_assets_parser.set_defaults(func=_handle_models_assets)
