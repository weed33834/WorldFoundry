#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_TARGETS = (
    REPO_ROOT / "worldfoundry" / "evaluation",
    REPO_ROOT / "scripts",
    REPO_ROOT / "test" / "eval_core",
)
RUFF_TARGETS = (
    REPO_ROOT / "worldfoundry" / "cli",
    REPO_ROOT / "worldfoundry" / "evaluation",
    REPO_ROOT / "worldfoundry" / "mcp",
    REPO_ROOT / "worldfoundry" / "runtime",
    REPO_ROOT / "scripts" / "benchmark_zoo",
    REPO_ROOT / "scripts" / "dev",
    REPO_ROOT / "scripts" / "docs",
    REPO_ROOT / "scripts" / "model_zoo",
    REPO_ROOT / "scripts" / "tools",
    REPO_ROOT / "test" / "eval_core",
)
SHELL_TARGETS = (REPO_ROOT / "scripts" / "setup",)
JSON_TARGETS = (
    REPO_ROOT / "worldfoundry" / "data",
    REPO_ROOT / "docs" / "fumadocs" / "content",
    REPO_ROOT / "pyproject.toml",
)
YAML_TARGETS = (
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks",
    REPO_ROOT / "environment.yml",
)
JSONL_COMPAT_FILES = frozenset(
    {
        REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "assets" / "fetv" / "fetv_data.json",
        REPO_ROOT
        / "worldfoundry"
        / "data"
        / "benchmarks"
        / "assets"
        / "fetv"
        / "sampled_prompts_for_fid_fvd"
        / "prompts_gen.json",
        REPO_ROOT
        / "worldfoundry"
        / "data"
        / "benchmarks"
        / "assets"
        / "fetv"
        / "sampled_prompts_for_fid_fvd"
        / "prompts_real.json",
    }
)
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "cache",
        "checkpoints",
        "data_cache",
        "dist",
        "hfd_datasets",
        "node_modules",
        "out",
        "output",
        "outputs",
        "tmp",
    }
)


def iter_python_files(paths: tuple[Path, ...]) -> list[Path]:
    """
    Return sorted Python files under the development check targets.

    Args:
        paths: Repository paths that should be scanned recursively.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*.py") if child.is_file())
    return sorted(files)


def iter_files(paths: tuple[Path, ...], suffixes: tuple[str, ...]) -> list[Path]:
    """
    Return sorted files matching suffixes under the given repository paths.

    Args:
        paths: Repository files or directories to scan.
        suffixes: File suffixes that should be included.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix in suffixes:
            files.append(path)
        elif path.is_dir():
            for root, dirnames, filenames in os.walk(path):
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if dirname not in SKIP_DIR_NAMES and not dirname.startswith(".")
                ]
                root_path = Path(root)
                for filename in filenames:
                    child = root_path / filename
                    if child.suffix in suffixes:
                        files.append(child)
    return sorted(set(files))


def format_check() -> int:
    """
    Validate that Python development targets are syntactically parseable.

    Args:
        None: The target paths are defined by PYTHON_TARGETS.
    """
    warnings: list[str] = []
    for path in iter_python_files(PYTHON_TARGETS):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line.rstrip() != line:
                warnings.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: trailing whitespace")
            if line.startswith("\t"):
                warnings.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: leading tab indentation")
    if warnings:
        print("\n".join(f"warning: {warning}" for warning in warnings))
    print("format-check: parsed Python targets successfully")
    return 0


def shell_check() -> int:
    """
    Validate setup shell scripts with bash syntax checking.

    Args:
        None: The setup script paths are defined by SHELL_TARGETS.
    """
    scripts = iter_files(SHELL_TARGETS, (".sh",))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(f"shell-check: validated {len(scripts)} setup shell scripts")
    return 0


def data_check() -> int:
    """
    Validate lightweight JSON and YAML files used by CI and manifests.

    Args:
        None: The structured data paths are defined by JSON_TARGETS and YAML_TARGETS.
    """
    json_files = [path for path in iter_files(JSON_TARGETS, (".json",)) if "node_modules" not in path.parts]
    yaml_files = iter_files(YAML_TARGETS, (".yml", ".yaml"))
    jsonl_compat_files = 0
    for path in json_files:
        source = path.read_text(encoding="utf-8")
        if path in JSONL_COMPAT_FILES:
            for line_number, line in enumerate(source.splitlines(), start=1):
                if line.strip():
                    json.loads(line)
                else:
                    raise ValueError(f"{path.relative_to(REPO_ROOT)}:{line_number}: blank JSONL line")
            jsonl_compat_files += 1
        else:
            json.loads(source)
    for path in yaml_files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    print(
        f"data-check: parsed {len(json_files)} JSON files"
        f" ({jsonl_compat_files} JSONL-compatible official assets)"
        f" and {len(yaml_files)} YAML files"
    )
    return 0


def ruff_check() -> int:
    """
    Run ruff over first-party CLI, evaluation, tooling, and eval-core tests.

    Args:
        None: The ruff target paths are repository-local constants.
    """
    targets = [path for path in RUFF_TARGETS if path.exists()]
    if not targets:
        print("ruff-check: no configured targets exist")
        return 0
    if importlib.util.find_spec("ruff") is None:
        print("ruff-check: skipped because ruff is not installed; run `make install-dev` first")
        return 0
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *[str(path) for path in targets]],
        cwd=REPO_ROOT,
        check=True,
    )
    print(f"ruff-check: validated {len(targets)} configured targets")
    return 0


def runtime_registry_check() -> int:
    """
    Validate runtime profile manifests and pipeline alias references.

    Args:
        None: Uses the canonical runtime profile tree under data/models/runtime.
    """
    from worldfoundry.evaluation.models.runtime.validate import validate_runtime_registry

    issues = validate_runtime_registry()
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        print("runtime-registry-check: failed")
        for issue in errors:
            location = f"{issue.field}: " if issue.field else ""
            print(f"  error [{issue.code}] {location}{issue.message}")
        return 1
    print(f"runtime-registry-check: validated runtime registry ({len(issues)} non-error findings)")
    return 0


def lint() -> int:
    """
    Run the default repository-local quality gate.

    Args:
        None: The check targets are repository-local constants.
    """
    format_check()
    shell_check()
    data_check()
    if runtime_registry_check() != 0:
        return 1
    return 0


def print_help_targets() -> int:
    """
    Print the repository development command surface.

    Args:
        None: Command descriptions are static and mirror the Makefile targets.
    """
    print(
        "\n".join(
            [
                "WorldFoundry development targets:",
                "  make install-core      Install editable package with core runtime dependencies.",
                "  make install-dev       Install editable package plus lightweight test dependencies.",
                "  make test-fast         Run the fast eval-core marker set.",
                "  make test-eval-core    Run deterministic eval-core contract tests.",
                "  make test-ux           Run CLI UX, catalog, config, and quickstart tests.",
                "  make docs-check        Validate documented CLI entrypoints and help surfaces.",
                "  make lint              Run Python syntax, setup shell, JSON, YAML, and runtime registry checks.",
                "  make ruff-check        Run explicit ruff checks over first-party eval, CLI, tooling, and tests.",
                "  make format-check      Parse Python development targets for syntax regressions.",
                "  make shell-check       Run bash -n over setup scripts.",
                "  make data-check        Parse repository JSON and YAML gate targets.",
                "  make compile-eval      Compile evaluation package and scripts.",
                "  make cli-check         Run the source-tree CLI package contract check.",
                "  make precommit         Run all pre-commit hooks over the repository.",
                "  make precommit-install Install the repository pre-commit hook.",
                "  make preflight         Run public runtime preflight before dataset staging.",
            ]
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldFoundry lightweight development command helpers.")
    parser.add_argument("--format-check", action="store_true", help="Parse development Python targets.")
    parser.add_argument("--ruff-check", action="store_true", help="Run ruff over first-party development targets.")
    parser.add_argument("--shell-check", action="store_true", help="Run bash -n over setup scripts.")
    parser.add_argument("--data-check", action="store_true", help="Parse repository JSON and YAML gate targets.")
    parser.add_argument("--runtime-registry-check", action="store_true", help="Validate runtime profile registry references.")
    parser.add_argument("--lint", action="store_true", help="Run all lightweight quality gates.")
    parser.add_argument("--help-targets", action="store_true", help="Print Makefile target help.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.help_targets:
        return print_help_targets()
    if args.lint:
        return lint()
    if args.ruff_check:
        return ruff_check()
    if args.format_check:
        return format_check()
    if args.shell_check:
        return shell_check()
    if args.data_check:
        return data_check()
    if args.runtime_registry_check:
        return runtime_registry_check()
    return print_help_targets()


if __name__ == "__main__":
    raise SystemExit(main())
