"""Discover persisted geometry assets eligible for Splats versus generic point previews."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Sequence

VIDEO_REPLAY_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ACTION_TRACE_EXTS = {".json", ".jsonl", ".npz", ".npy", ".pkl"}
EPISODE_METADATA_EXTS = {".json", ".jsonl", ".yaml", ".yml", ".toml"}


def _npz_contains_geometry(path: Path) -> bool:
    """Return whether an NPZ has explicit XYZ or a depth camera bundle."""

    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
    except Exception:
        return False
    point_keys = {"world_points", "points", "xyz", "point_cloud", "pts3d", "point_map"}
    return bool(keys & point_keys) or {"depth", "intrinsics", "extrinsics"} <= keys


def normalize_output_relative(path_text: str, output_dir: str | Path) -> str:
    """Return ``path_text`` trimmed; if absolute under ``output_dir`` make it posix relative."""

    text = Path(path_text).as_posix()
    outp = Path(output_dir).resolve()
    maybe = Path(text)
    if maybe.is_absolute():
        try:
            rel = maybe.resolve().relative_to(outp)
            return rel.as_posix()
        except ValueError:
            return text
    return text.strip().lstrip("./")


def first_geometry_point_candidate(
    artifact_paths: Sequence[str],
    output_dir: str | Path,
    *,
    gs_ply_predicate,
) -> str | None:
    """Pick the first on-disk generic point/depth bundle eligible for Viser."""

    ordered = sorted({str(p) for p in artifact_paths if p})
    for raw in ordered:
        path = Path(raw)
        if not path.exists():
            continue
        suf = path.suffix.lower()
        if suf == ".ply":
            if gs_ply_predicate(path):
                continue
            return normalize_output_relative(str(path.resolve()), output_dir)
        if suf in {".pcd", ".xyz"}:
            return normalize_output_relative(str(path.resolve()), output_dir)
        if suf == ".npz" and _npz_contains_geometry(path):
            return normalize_output_relative(str(path.resolve()), output_dir)
    scanned = sorted(Path(output_dir).glob("**/*.ply"))
    for cand in scanned:
        if cand.is_file() and not gs_ply_predicate(cand):
            return cand.relative_to(Path(output_dir).resolve()).as_posix()
    return None


def first_splat_asset(artifact_paths: Iterable[str], *, gs_ply_predicate=None) -> tuple[str | None, str | None]:
    """Return `(path_text, format_hint)` prioritizing Gaussian-friendly extensions."""

    exts_priority = (
        ".spz",
        ".ksplat",
        ".splat",
        ".sog",
    )
    seen = sorted({Path(p).resolve() for p in artifact_paths if p})
    buckets: dict[str, Path | None] = {ext.strip("."): None for ext in exts_priority}
    gaussian_ply: Path | None = None
    for path in seen:
        if not path.exists() or path.is_dir():
            continue
        suf = path.suffix.lower()
        if suf == ".ply" and gs_ply_predicate is not None and gaussian_ply is None and gs_ply_predicate(path):
            gaussian_ply = path
            continue
        fmt = suf.strip(".") or None
        if fmt in buckets and buckets[fmt] is None:
            buckets[fmt] = path

    ordered_keys = tuple(ext.strip(".") for ext in exts_priority)
    for key in ordered_keys:
        resolved = buckets.get(key)
        if resolved:
            return resolved.as_posix(), key
    if gaussian_ply is not None:
        return gaussian_ply.as_posix(), "ply_gaussian"
    return None, None


def first_embodied_trace_candidate(artifact_paths: Sequence[str], output_dir: str | Path) -> str | None:
    """Pick the first action/policy trace artifact for embodied and visual-action models."""

    priority_terms = (
        "action_trace",
        "action-trace",
        "actions",
        "robot_action",
        "policy",
        "trajectory",
        "rollout",
        "latent_action",
        "tokens",
    )
    for path in _ordered_named_candidates(artifact_paths, priority_terms, ACTION_TRACE_EXTS):
        if _is_empty_json_trace(path):
            continue
        return normalize_output_relative(path.as_posix(), output_dir)
    return None


def _is_empty_json_trace(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload == [] or payload == {}


def first_simulator_replay_candidate(artifact_paths: Sequence[str], output_dir: str | Path) -> str | None:
    """Pick a video-like simulator or robot replay artifact when one is present."""

    priority_terms = (
        "sim",
        "simulator",
        "rollout",
        "episode",
        "robot",
        "policy",
        "env",
        "replay",
        "trajectory",
    )
    for path in _ordered_named_candidates(artifact_paths, priority_terms, VIDEO_REPLAY_EXTS):
        return normalize_output_relative(path.as_posix(), output_dir)
    return None


def first_episode_metadata_candidate(artifact_paths: Sequence[str], output_dir: str | Path) -> str | None:
    """Pick environment or episode metadata for simulator-oriented inspection."""

    priority_terms = (
        "episode",
        "sim",
        "simulator",
        "environment",
        "env",
        "metadata",
        "task",
    )
    for path in _ordered_named_candidates(artifact_paths, priority_terms, EPISODE_METADATA_EXTS):
        return normalize_output_relative(path.as_posix(), output_dir)
    return None


def _ordered_named_candidates(
    artifact_paths: Sequence[str],
    priority_terms: Sequence[str],
    suffixes: set[str],
) -> list[Path]:
    """Return existing artifact paths whose names match a semantic priority list."""

    candidates: list[tuple[int, int, str, Path]] = []
    for raw in sorted({str(p) for p in artifact_paths if p}):
        path = Path(raw)
        if not path.exists() or path.is_dir() or path.suffix.lower() not in suffixes:
            continue
        lowered = path.name.lower()
        matched_index = next(
            (index for index, term in enumerate(priority_terms) if term in lowered),
            None,
        )
        if matched_index is None:
            continue
        candidates.append((matched_index, len(path.parts), lowered, path.resolve()))
    return [path for _, _, _, path in sorted(candidates)]



class RunRecordProvider:
    """Convert run records or artifact directories into lightweight scenes."""

    provider_id = "run_record"

    def discover(self, source):
        from pathlib import Path
        from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact

        output_dir = getattr(source, "output_dir", None) or getattr(source, "path", None) or source
        if output_dir is None:
            return None
        root = Path(output_dir)
        if not root.exists():
            return None
        layers = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            artifact = infer_visualization_artifact(path)
            if artifact.kind == "artifact":
                continue
            layer_id = path.relative_to(root).as_posix().replace("/", ":")
            layers.append(Layer(layer_id=layer_id, kind=artifact.kind, uri=path.as_posix(), metadata={"format": artifact.format_hint}))
        if not layers:
            return None
        return VisualizationScene(scene_id=f"run/{root.name}", title=root.name, layers=tuple(layers), metadata={"source": root.as_posix()})
