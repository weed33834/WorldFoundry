"""Physics-IQ prompt materialization and generated-video layout helpers."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.utils import write_jsonl
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "physics-iq"
DESCRIPTIONS_REL = Path("descriptions/descriptions.csv")
BENCHMARK_DIR_NAME = "physics-IQ-benchmark"

CANONICAL_PROMPT_COUNT = 198
VIEWS = ("perspective-left", "perspective-center", "perspective-right")
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def resolve_physics_iq_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_PHYSICS_IQ_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_descriptions_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Physics-IQ descriptions file not found: {path}")
        return path
    env_descriptions = _env_path("WORLDFOUNDRY_PHYSICS_IQ_DESCRIPTIONS")
    if env_descriptions is not None:
        if not env_descriptions.is_file():
            raise FileNotFoundError(f"Physics-IQ descriptions file not found: {env_descriptions}")
        return env_descriptions
    bundled = bundled_benchmark_asset(BENCHMARK_ID, DESCRIPTIONS_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_physics_iq_root()
    if root is None:
        raise FileNotFoundError(
            "Physics-IQ descriptions file is missing. Set WORLDFOUNDRY_PHYSICS_IQ_DESCRIPTIONS "
            "or WORLDFOUNDRY_PHYSICS_IQ_ROOT."
        )
    candidate = root / DESCRIPTIONS_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"Physics-IQ descriptions file not found: {candidate}")
    return candidate


def _take_one(row: dict[str, str]) -> bool:
    scenario = str(row.get("scenario") or "")
    return "_take-1_" in scenario


def load_description_rows(*, descriptions_path: Path | None = None) -> list[dict[str, str]]:
    path = resolve_descriptions_path(explicit=descriptions_path)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not _take_one(row):
                continue
            generated_name = str(row.get("generated_video_name") or "").strip()
            description = str(row.get("description") or "").strip()
            category = str(row.get("category") or "").strip()
            scenario = str(row.get("scenario") or "").strip()
            if not generated_name or not description:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "description": description,
                    "category": category,
                    "generated_video_name": generated_name,
                }
            )
    if not rows:
        raise ValueError(f"Physics-IQ descriptions are empty after take-1 filtering: {path}")
    return rows


def unique_generation_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for row in rows:
        video_name = row["generated_video_name"]
        if video_name in seen:
            continue
        seen.add(video_name)
        records.append(row)
    return sorted(records, key=lambda item: item["generated_video_name"])


def video_stem_for_record(record: dict[str, str]) -> str:
    name = record["generated_video_name"]
    return Path(name).stem


def video_filename_for_record(record: dict[str, str]) -> str:
    name = record["generated_video_name"]
    return name if name.endswith(".mp4") else f"{name}.mp4"


def materialize_physics_iq_generation_requests(
    *,
    limit: int | None = None,
    descriptions_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_generation_records(load_description_rows(descriptions_path=descriptions_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = video_stem_for_record(record)
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="physics-iq",
                split=split,
                inputs={
                    "prompt": record["description"],
                    "prompt_id": sample_id,
                    "generation_text": record["description"],
                    "category": record["category"],
                    "generated_video_name": record["generated_video_name"],
                    "scenario": record["scenario"],
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_physics_iq_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
) -> tuple[int, int]:
    """Copy model outputs into official ``generated_video_name`` filenames."""
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    materialized = 0
    placeholders = 0
    manifest_rows: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in generation_output_dir.iterdir() if path.is_dir()):
        result_path = sample_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        sample_id = str(payload.get("sample_id") or sample_dir.name)
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        source = outputs.get(output_artifact) or outputs.get("generated_video")
        if not source:
            continue
        source_path = Path(str(source))
        if not source_path.is_file():
            continue
        target_name = sample_id if sample_id.endswith(".mp4") else f"{sample_id}.mp4"
        target_path = generated_artifact_dir / target_name
        shutil.copy2(source_path, target_path)
        materialized += 1
        manifest_rows.append({"sample_id": sample_id, "artifact": output_artifact, "path": str(target_path)})

    results_path = generation_output_dir / "results.jsonl"
    if results_path.is_file():
        for result in (GenerationResult.from_dict(row) for row in _read_jsonl(results_path)):
            artifact = result.artifacts.get(output_artifact) or result.artifacts.get("generated_video")
            if artifact is None:
                continue
            from worldfoundry.evaluation.utils import local_path_for_uri

            source_path = local_path_for_uri(str(artifact))
            if source_path is None or not source_path.is_file():
                continue
            target_name = (
                result.sample_id if result.sample_id.endswith(".mp4") else f"{result.sample_id}.mp4"
            )
            target_path = generated_artifact_dir / target_name
            if target_path.is_file():
                continue
            shutil.copy2(source_path, target_path)
            materialized += 1
            manifest_rows.append(
                {"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)}
            )

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders
