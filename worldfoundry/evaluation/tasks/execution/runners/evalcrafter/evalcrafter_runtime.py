"""EvalCrafter scorer runtime for WorldFoundry-generated artifacts."""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
    CANONICAL_PROMPT_COUNT,
    EXPECTED_VIDEO_COUNT,
    load_prompt_records,
    resolve_evalcrafter_root,
    resolve_prompt700_path,
    unique_prompt_records,
)

def latest_final_result(results_dir: Path) -> Path:
    if results_dir.is_file():
        return results_dir
    direct = results_dir / "final_result.txt"
    if direct.is_file():
        return direct
    candidates = sorted(results_dir.glob("*final_result*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"EvalCrafter results must contain final_result.txt: {results_dir}")

VIDEO_SUFFIXES = frozenset({".mp4"})


@dataclass(frozen=True)
class EvalCrafterScorerConfig:
    backend: str
    repo_root: Path | None
    strict: bool = False


def scorer_config_from_env() -> EvalCrafterScorerConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_EVALCRAFTER_SCORER_BACKEND")
        or os.environ.get("WORLDFOUNDRY_EVALCRAFTER_RUNTIME_BACKEND")
        or "artifact"
    ).strip().lower()
    strict = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return EvalCrafterScorerConfig(
        backend=backend,
        repo_root=resolve_evalcrafter_root(),
        strict=strict,
    )


def _matching_videos(*, generated_artifact_dir: Path, prompt_ids: set[str]) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    matched: list[str] = []
    for path in sorted(generated_artifact_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in prompt_ids:
            matched.append(path.stem)
    return matched


def _env_results_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else None


def _artifact_results_path(*, generated_artifact_dir: Path, output_dir: Path) -> Path | None:
    for candidate in (
        _env_results_path(),
        generated_artifact_dir / "final_result.txt",
        generated_artifact_dir / "evalcrafter_final_result.txt",
        output_dir / "final_result.txt",
    ):
        if candidate is None:
            continue
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            try:
                return latest_final_result(candidate)
            except FileNotFoundError:
                pass
    return None


def _deterministic_mock_score(*, seed: str, offset: int, low: float = 0.55, span: float = 0.3) -> float:
    digest = hashlib.sha256(f"{seed}:{offset}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return round(low + (bucket / 999.0) * span, 4)


def _write_mock_final_result(*, output_path: Path, seed: str) -> Path:
    raw_metrics = {
        "VQA_A": _deterministic_mock_score(seed=seed, offset=1),
        "VQA_T": _deterministic_mock_score(seed=seed, offset=2),
        "IS": round(8.0 + _deterministic_mock_score(seed=seed, offset=3, low=0.0, span=6.0), 4),
        "clip_temp_score": _deterministic_mock_score(seed=seed, offset=4),
        "warping_error": _deterministic_mock_score(seed=seed, offset=5, low=0.01, span=0.05),
        "face_consistency_score": _deterministic_mock_score(seed=seed, offset=6),
        "action_score": _deterministic_mock_score(seed=seed, offset=7),
        "motion_ac_score": _deterministic_mock_score(seed=seed, offset=8),
        "flow_score": _deterministic_mock_score(seed=seed, offset=9),
        "clip_score": _deterministic_mock_score(seed=seed, offset=10, low=0.25, span=0.4),
        "blip_bleu": _deterministic_mock_score(seed=seed, offset=11, low=0.25, span=0.4),
        "sd_score": _deterministic_mock_score(seed=seed, offset=12),
        "detection_score": _deterministic_mock_score(seed=seed, offset=13),
        "color_score": _deterministic_mock_score(seed=seed, offset=14),
        "count_score": _deterministic_mock_score(seed=seed, offset=15),
        "ocr_score": _deterministic_mock_score(seed=seed, offset=16, low=0.01, span=0.08),
        "celebrity_id_score": _deterministic_mock_score(seed=seed, offset=17, low=0.01, span=0.08),
    }
    visual_quality = round((raw_metrics["VQA_A"] + raw_metrics["VQA_T"]) / 2.0, 4)
    text_video_alignment = round(
        (
            raw_metrics["clip_score"]
            + raw_metrics["blip_bleu"]
            + raw_metrics["sd_score"]
            + raw_metrics["detection_score"]
            + raw_metrics["color_score"]
            + raw_metrics["count_score"]
        )
        / 6.0,
        4,
    )
    motion_quality = round(
        (raw_metrics["action_score"] + raw_metrics["motion_ac_score"] + raw_metrics["flow_score"]) / 3.0,
        4,
    )
    temporal_consistency = round(
        (raw_metrics["clip_temp_score"] + raw_metrics["face_consistency_score"] + (1.0 - raw_metrics["warping_error"])) / 3.0,
        4,
    )
    total = round((visual_quality + text_video_alignment + motion_quality + temporal_consistency) / 4.0, 4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"Metrics: {raw_metrics!r}",
                f"Visual Quality {visual_quality}",
                f"Text-Video Alignment {text_video_alignment}",
                f"Motion Quality {motion_quality}",
                f"Temporal Consistency {temporal_consistency}",
                f"Total {total}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def run_evalcrafter_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: EvalCrafterScorerConfig,
    prompt700_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "final_result.txt"
    prompt_records = unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_ids = {record["prompt_id"] for record in prompt_records}
    matched_videos = _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)

    if config.backend == "mock":
        seed = f"evalcrafter-mock:{len(matched_videos)}:{limit or 'all'}"
        _write_mock_final_result(output_path=results_path, seed=seed)
        return {
            "backend": "mock",
            "results_path": str(results_path.resolve()),
            "video_count": len(matched_videos),
            "prompt_count": len(prompt_records),
        }

    if config.backend not in {"artifact", "worldfoundry"}:
        raise ValueError(
            "EvalCrafter runner no longer launches benchmark-local shell scripts. "
            "Run metrics through WorldFoundry/base-model infrastructure and pass "
            "final_result.txt via WORLDFOUNDRY_EVALCRAFTER_RESULTS_PATH or --generated-artifact-dir."
        )
    source_results = _artifact_results_path(generated_artifact_dir=generated_artifact_dir, output_dir=output_dir)
    if source_results is None:
        raise FileNotFoundError(
            "EvalCrafter artifact evaluation requires final_result.txt. Set "
            "WORLDFOUNDRY_EVALCRAFTER_RESULTS_PATH or place final_result.txt under --generated-artifact-dir."
        )
    if source_results.resolve() != results_path.resolve():
        results_path.write_bytes(source_results.read_bytes())
    return {
        "backend": "artifact",
        "results_path": str(results_path.resolve()),
        "source_results_path": str(source_results.resolve()),
        "video_count": len(matched_videos),
        "prompt_count": len(prompt_records),
    }


def validate_official_inputs(evalcrafter_root: Path, videos_dir: Path) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import resolve_prompt700_path

    prompt_path = resolve_prompt700_path(repo_root=evalcrafter_root)
    prompt_lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    prompt = {
        "path": str(prompt_path),
        "exists": prompt_path.is_file(),
        "line_count": len(prompt_lines),
        "expected_line_count": CANONICAL_PROMPT_COUNT,
    }
    prompt["ok"] = prompt["exists"] and prompt["line_count"] == CANONICAL_PROMPT_COUNT

    expected_names = {f"{index:04d}.mp4" for index in range(EXPECTED_VIDEO_COUNT)}
    mp4_names: set[str] = set()
    non_video_entries: list[str] = []
    subdirectories: list[str] = []
    if videos_dir.is_dir():
        for path in sorted(videos_dir.iterdir()):
            if path.is_dir():
                subdirectories.append(path.name)
            elif path.is_file() and path.suffix.lower() == ".mp4":
                mp4_names.add(path.name)
            else:
                non_video_entries.append(path.name)

    missing = sorted(expected_names - mp4_names)
    unexpected = sorted(mp4_names - expected_names)
    candidate_video_dirs: list[dict[str, Any]] = []
    if videos_dir.is_dir():
        for candidate in sorted(path for path in videos_dir.rglob("*") if path.is_dir()):
            direct_mp4_count = sum(1 for item in candidate.iterdir() if item.is_file() and item.suffix.lower() == ".mp4")
            if direct_mp4_count:
                candidate_video_dirs.append({"path": str(candidate), "mp4_count": direct_mp4_count})
    videos = {
        "path": str(videos_dir),
        "exists": videos_dir.is_dir(),
        "mp4_count": len(mp4_names) if videos_dir.is_dir() else None,
        "expected_mp4_count": EXPECTED_VIDEO_COUNT,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "unexpected_count": len(unexpected),
        "unexpected_examples": unexpected[:20],
        "non_video_entry_count": len(non_video_entries),
        "subdirectory_count": len(subdirectories),
        "candidate_video_dirs": candidate_video_dirs[:20],
    }
    videos["ok"] = (
        videos["exists"]
        and len(mp4_names) == EXPECTED_VIDEO_COUNT
        and not missing
        and not unexpected
        and not non_video_entries
        and not subdirectories
    )

    result = {
        "ok": bool(prompt["ok"] and videos["ok"]),
        "prompt": prompt,
        "videos": videos,
    }
    reasons: list[str] = []
    if not prompt["ok"]:
        reasons.append(f"prompt700.txt must contain exactly {CANONICAL_PROMPT_COUNT} prompts")
    if not videos["ok"]:
        reasons.append(
            f"videos-dir must contain exactly {EXPECTED_VIDEO_COUNT} direct files named 0000.mp4 through 0699.mp4"
        )
    result["reasons"] = reasons
    return result
