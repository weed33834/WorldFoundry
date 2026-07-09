"""iWorld-Bench prompt materialization from released dataset metadata CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "iworld-bench"
IN_TREE_IWORLD_BENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "iworldbench"
METADATA_REL = Path("dataset/all_pack/metadata.csv")
CAMERA_FOLLOWING_METADATA_REL = Path("dataset/all_pack/camera_following_metadata.csv")
CANONICAL_PROMPT_COUNT = 4900

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def resolve_iworldbench_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_IWORLD_BENCH_ROOT"),
        IN_TREE_IWORLD_BENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_dataset_root(*, explicit: Path | None = None, repo_root: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT"),
        repo_root,
        resolve_iworldbench_root(),
    ):
        if candidate is not None and candidate.is_dir():
            metadata = candidate / METADATA_REL
            if metadata.is_file():
                return candidate.expanduser().resolve()
    bundled = bundled_benchmark_assets_root(BENCHMARK_ID)
    if (bundled / METADATA_REL).is_file():
        return bundled
    return None


def resolve_metadata_csv_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
    split: str = "diff",
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"iWorld-Bench metadata CSV not found: {path}")
        return path
    env_manifest = _env_path("WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"iWorld-Bench metadata CSV not found: {env_manifest}")
        return env_manifest
    relative = CAMERA_FOLLOWING_METADATA_REL if split.lower() == "camera_following" else METADATA_REL
    bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
    if bundled.is_file():
        return bundled
    dataset_root = resolve_dataset_root(repo_root=repo_root)
    if dataset_root is None:
        raise FileNotFoundError(
            "iWorld-Bench metadata CSV is missing. Set WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT, "
            "WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST, or WORLDFOUNDRY_IWORLD_BENCH_ROOT."
        )
    candidate = dataset_root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"iWorld-Bench metadata CSV not found: {candidate}")
    return candidate


def _generation_text(row: dict[str, Any]) -> str:
    for key in ("text_description", "prompt", "caption", "action_text", "description"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _prompt_id_from_row(row: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "video_id", "prompt_id", "index", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(index)


def load_prompt_records(
    *,
    meta_csv_path: Path | None = None,
    split: str = "diff",
) -> list[dict[str, Any]]:
    path = resolve_metadata_csv_path(explicit=meta_csv_path, split=split)
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if not row:
                continue
            prompt_id = _prompt_id_from_row(row, index)
            prompt = _generation_text(row)
            if not prompt_id:
                continue
            records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt or f"iworld-bench:{split}:{prompt_id}",
                    "split": split,
                    "first_frame_path": row.get("first_frame") or row.get("image_path") or row.get("asset_path"),
                    "video_path": row.get("video_path") or row.get("output_path"),
                    "raw": dict(row),
                }
            )
    if not records:
        raise ValueError(f"iWorld-Bench prompt records are empty after validation: {path}")
    return records


def unique_prompt_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return records


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    video_path = str(record.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).name
    return f"{record['prompt_id']}.mp4"


def materialize_iworldbench_generation_requests(
    *,
    limit: int | None = None,
    meta_csv_path: Path | None = None,
    split: str = "diff",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(meta_csv_path=meta_csv_path, split=split))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="iworld-bench",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "first_frame_path": record.get("first_frame_path"),
                    "official_video_name": official_video_filename_for_record(record),
                    "split": split,
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)
