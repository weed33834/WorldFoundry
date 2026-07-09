"""Physics-IQ artifact/result importer for WorldFoundry-generated evaluations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_metrics import METRIC_ORDER
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
    resolve_descriptions_path,
    resolve_physics_iq_root,
)


@dataclass(frozen=True)
class PhysicsIQScorerConfig:
    backend: str
    descriptions_path: Path | None
    strict: bool = False


def scorer_config_from_env() -> PhysicsIQScorerConfig:
    backend = os.environ.get("WORLDFOUNDRY_PHYSICS_IQ_SCORER_BACKEND", "artifact").strip().lower()
    repo_root = resolve_physics_iq_root()
    descriptions_path = None
    try:
        descriptions_path = resolve_descriptions_path(repo_root=repo_root)
    except FileNotFoundError:
        descriptions_path = None
    strict = os.environ.get("WORLDFOUNDRY_PHYSICS_IQ_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return PhysicsIQScorerConfig(
        backend=backend,
        descriptions_path=descriptions_path,
        strict=strict,
    )


def discover_physics_iq_results(search_roots: list[Path]) -> Path | None:
    globs = (
        "results_summary.csv",
        "physics_iq_results*.csv",
        "physics_iq_results*.json",
        "*physics_iq*.csv",
        "*physics-iq*.csv",
        "*physics_iq*.json",
        "*physics-iq*.json",
    )
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in {".csv", ".json"}:
            return root
        for pattern in globs:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[-1]
    return None


def _env_results_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def _artifact_results_path(*, generated_artifact_dir: Path | None, output_dir: Path) -> Path | None:
    search_roots = [output_dir]
    if generated_artifact_dir is not None:
        search_roots.insert(0, generated_artifact_dir)
    for candidate in (_env_results_path(), discover_physics_iq_results(search_roots)):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _copy_artifact_results(*, source_path: Path, output_dir: Path) -> Path:
    suffix = source_path.suffix if source_path.suffix else ".csv"
    output_path = output_dir / f"physics_iq_results{suffix}"
    output_path.write_bytes(source_path.read_bytes())
    return output_path


def run_physics_iq_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: PhysicsIQScorerConfig,
    descriptions_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = descriptions_path or config.descriptions_path
    video_count = sum(
        1
        for path in generated_artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    ) if generated_artifact_dir.is_dir() else 0
    if config.backend not in {"artifact", "worldfoundry", "official"}:
        raise ValueError(
            "Unsupported Physics-IQ scorer backend "
            f"{config.backend!r}. Use 'artifact' to import results produced by the "
            "WorldFoundry evaluation pipeline."
        )
    source_path = _artifact_results_path(
        generated_artifact_dir=generated_artifact_dir,
        output_dir=output_dir,
    )
    if source_path is None:
        raise FileNotFoundError(
            "Physics-IQ artifact evaluation requires an existing CSV/JSON results file. "
            "Set WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH or place physics_iq_results*.csv/json "
            "under --generated-artifact-dir."
        )
    results_path = _copy_artifact_results(source_path=source_path, output_dir=output_dir)
    return {
        "backend": "artifact",
        "results_path": str(results_path.resolve()),
        "results_csv": str(results_path.resolve()) if results_path.suffix.lower() == ".csv" else None,
        "source_results_path": str(source_path.resolve()),
        "video_count": video_count,
        "descriptions_path": None if descriptions is None else str(descriptions),
        "metric_ids": list(METRIC_ORDER),
    }
