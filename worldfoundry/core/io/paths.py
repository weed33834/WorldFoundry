"""WorldFoundry Path and Directory Resolution Engine.

This module provides unified, robust path resolution utilities that completely decouple the
WorldFoundry runtime and evaluation benchmarks from host-specific, hardcoded absolute directory paths.

By using logical placeholders and environment-variable expansion tokens (e.g., `${WORLDFOUNDRY_CACHE_DIR}`),
this engine guarantees:
1. Portability: Evaluator runners, dataset manifests, and experiment checkpoints can run unmodified
   across different systems (e.g., Aliyun PAI DLC jobs, local workstations, or Docker containers).
2. Predictability: Explicit fallback patterns resolve logical paths to project-relative paths (e.g. searching
   upward for a `pyproject.toml` descriptor) when specific environment variables are missing.
3. Hermetic Cache Boundaries: Prevents external caches or standard system temp folders from being contaminated
   unless explicitly overridden.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping


def package_root() -> Path:
    """Returns the resolved absolute path of the installed `worldfoundry` package root."""
    return Path(__file__).resolve().parents[2]


def package_module_root(package: str) -> Path:
    """Resolve the source directory for an importable package."""

    spec = find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Could not resolve package: {package}")
    return Path(spec.origin).resolve().parent


def project_root(start: str | Path | None = None) -> Path:
    """Walks upward from a starting path to locate the root repository containing `pyproject.toml`.

    This helper provides robust local development support, falling back to a package-relative
    root if executed from a system-wide python site-packages deployment.
    """
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return package_root().parents[1]


def worldfoundry_path_tokens(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Generates the dictionary of logical path-token replacements used across the system.

    Builds dynamic mappings for artifact, checkpoint, data, conda, and repo paths.
    Prioritizes explicit environment overrides (such as `WORLDFOUNDRY_HOME` or `WORLDFOUNDRY_CACHE_DIR`)
    and falls back to user-home cache directories when variables are unset.
    """
    environ = dict(os.environ if env is None else env)
    root = project_root()
    package = package_root()
    project_parent = root.parent
    adjacent_ckpt = project_parent / "ckpt"
    adjacent_conda = project_parent / "conda"
    adjacent_conda_envs = adjacent_conda / "envs"
    explicit_home = environ.get("WORLDFOUNDRY_HOME")
    home = Path(explicit_home or Path.home() / ".cache" / "worldfoundry").expanduser()
    cache_default = home / "cache" if explicit_home else home
    cache = Path(environ.get("WORLDFOUNDRY_CACHE_DIR") or cache_default).expanduser()
    data_dir = Path(
        environ.get("WORLDFOUNDRY_DATA_DIR")
        or environ.get("WORLDFOUNDRY_BENCHMARK_DATA_ROOT")
        or (home / "data" if explicit_home else cache / "data")
    ).expanduser()
    artifact_dir = Path(
        environ.get("WORLDFOUNDRY_ARTIFACT_DIR")
        or environ.get("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
        or (home / "artifacts" if explicit_home else cache / "artifacts")
    ).expanduser()
    model_dir = Path(
        environ.get("WORLDFOUNDRY_MODEL_DIR") or (home / "models" if explicit_home else cache / "models")
    ).expanduser()
    default_model_source = cache / "official_runtime_repos"
    model_source = Path(environ.get("WORLDFOUNDRY_MODEL_SOURCE_DIR") or default_model_source).expanduser()
    default_ckpt_dir = adjacent_ckpt if adjacent_ckpt.is_dir() else (
        home / "checkpoints" if explicit_home else cache / "checkpoints"
    )
    ckpt_dir = Path(environ.get("WORLDFOUNDRY_CKPT_DIR") or default_ckpt_dir).expanduser()
    hfd_root = Path(environ.get("WORLDFOUNDRY_HFD_ROOT") or ckpt_dir / "hfd").expanduser()
    default_conda_root = adjacent_conda if adjacent_conda.is_dir() else (
        home / "conda" if explicit_home else cache / "conda"
    )
    conda_root = Path(environ.get("WORLDFOUNDRY_CONDA_ROOT") or default_conda_root).expanduser()
    default_conda_envs_root = adjacent_conda_envs if adjacent_conda_envs.is_dir() else (
        home / "conda_envs" if explicit_home else cache / "conda_envs"
    )
    conda_envs_root = Path(
        environ.get("WORLDFOUNDRY_CONDA_ENVS_ROOT")
        or environ.get("WORLDFOUNDRY_CONDA_ENV_ROOT")
        or default_conda_envs_root
    ).expanduser()
    return {
        "WORLDFOUNDRY_REPO_ROOT": str(root),
        "WORLDFOUNDRY_PACKAGE_ROOT": str(package),
        "WORLDFOUNDRY_DATA_ROOT": str(package / "data"),
        "WORLDFOUNDRY_CACHE_DIR": str(cache),
        "WORLDFOUNDRY_HOME": str(home),
        "WORLDFOUNDRY_DATA_DIR": str(data_dir),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(artifact_dir),
        "WORLDFOUNDRY_MODEL_DIR": str(model_dir),
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": str(model_source),
        "WORLDFOUNDRY_CKPT_DIR": str(ckpt_dir),
        "WORLDFOUNDRY_HFD_ROOT": str(hfd_root),
        "WORLDFOUNDRY_CONDA_ROOT": str(conda_root),
        "WORLDFOUNDRY_CONDA_ENVS_ROOT": str(conda_envs_root),
    }


def resolve_worldfoundry_path(value: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expands structural WorldFoundry path tokens (e.g. `${WORLDFOUNDRY_CKPT_DIR}`) and home markers (~).

    Performs precise regex-free variable mapping replacement while preserving subfolder hierarchies.
    """
    replacements = worldfoundry_path_tokens(env)
    if env is not None:
        replacements.update(
            {name: str(replacement) for name, replacement in env.items() if name.startswith("WORLDFOUNDRY_")}
        )
    expanded = str(value)
    for name, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        expanded = expanded.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
    return Path(os.path.expandvars(expanded)).expanduser()


def official_runtime_repo_path(
    repo_name: str,
    *,
    specific_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolves the checkout path of an official repository or model library dependency.

    Prioritizes specific system environmental overrides (e.g. custom paths for particular repos)
    before applying standard model source resolutions.
    """
    environ = dict(os.environ if env is None else env)
    if specific_env and environ.get(specific_env):
        return resolve_worldfoundry_path(environ[specific_env], environ)
    if environ.get("WORLDFOUNDRY_GITHUB_REPOS_ROOT"):
        return resolve_worldfoundry_path(Path(environ["WORLDFOUNDRY_GITHUB_REPOS_ROOT"]) / repo_name, environ)
    return resolve_worldfoundry_path(Path("${WORLDFOUNDRY_MODEL_SOURCE_DIR}") / repo_name, environ)


def model_source_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory containing official third-party codebases and model packages."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_MODEL_SOURCE_DIR}", env)


def cache_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the standard WorldFoundry cached download directory."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CACHE_DIR}", env)


def local_data_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root location containing local datasets, evaluation splits, and physical assets."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_DATA_DIR}", env)


def local_model_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory containing local model weights, configs, and adapters."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_MODEL_DIR}", env)


def artifact_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory where run scorecards and generated log artifacts are serialized."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_ARTIFACT_DIR}", env)


def checkpoint_root_path(
    *parts: str | Path,
    specific_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolves a target model checkpoint directory path with nested subfolders.

    Avoids host-specific hardcoding by querying general and model-specific variables.
    """
    environ = dict(os.environ if env is None else env)
    if specific_env and environ.get(specific_env):
        root = resolve_worldfoundry_path(environ[specific_env], environ)
    else:
        root = resolve_worldfoundry_path("${WORLDFOUNDRY_CKPT_DIR}", environ)
    return root.joinpath(*(Path(part) for part in parts))


def hfd_root_path(*parts: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Resolves the hfd-style local downloader checkpoint directory."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_HFD_ROOT}", env).joinpath(*(Path(part) for part in parts))


def conda_envs_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root path containing model-specific python environments and dependencies."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CONDA_ENVS_ROOT}", env)


def conda_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the base installation folder of the system conda package manager."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CONDA_ROOT}", env)


def resolve_package_path(*parts: str | Path) -> Path:
    """Resolves a subpath relative to the active `worldfoundry` package source directory."""
    return package_root().joinpath(*(Path(part) for part in parts))


def resolve_data_path(*parts: str | Path) -> Path:
    """Resolves a subpath under the internal `worldfoundry/data` asset directory."""
    return resolve_package_path("data", *parts)


def repo_relative_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Squeezes a path relative to the active repository parent to keep logs clean and short.

    If the path is outside the repository tree, falls back gracefully to a fully resolved POSIX string.
    """
    resolved = Path(path).expanduser().resolve()
    repo = Path(root).expanduser().resolve() if root is not None else project_root()
    try:
        return resolved.relative_to(repo).as_posix()
    except ValueError:
        return resolved.as_posix()


__all__ = [
    "checkpoint_root_path",
    "artifact_root_path",
    "cache_root_path",
    "conda_envs_root_path",
    "conda_root_path",
    "hfd_root_path",
    "local_data_root_path",
    "local_model_root_path",
    "model_source_root_path",
    "package_module_root",
    "official_runtime_repo_path",
    "package_root",
    "project_root",
    "repo_relative_path",
    "resolve_data_path",
    "resolve_package_path",
    "resolve_worldfoundry_path",
    "worldfoundry_path_tokens",
]
