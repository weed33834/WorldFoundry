"""SceneFID protocol: FID on object crops (OC-GAN style)."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from worldfoundry.evaluation.tasks.metrics.fid.compute import compute_fid

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def _load_bboxes(path: Path) -> dict[str, list[list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bboxes JSON must be an object mapping image keys to bbox lists")
    normalized: dict[str, list[list[float]]] = {}
    for key, boxes in payload.items():
        if not isinstance(boxes, list):
            raise ValueError(f"bbox list for {key!r} must be a list")
        normalized[str(key)] = [list(map(float, box)) for box in boxes]
    return normalized


def _resolve_image_path(image_root: Path, key: str) -> Path:
    candidate = Path(key)
    if candidate.is_file():
        return candidate
    relative = image_root / key
    if relative.is_file():
        return relative
    by_name = image_root / candidate.name
    if by_name.is_file():
        return by_name
    raise FileNotFoundError(f"could not resolve image for bbox key {key!r} under {image_root}")


def _save_crop(image: Image.Image, box: Sequence[float], dest: Path, *, min_crop_size: int) -> bool:
    width, height = image.size
    if len(box) == 4:
        x1, y1, x2, y2 = box
    elif len(box) == 5:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
    else:
        raise ValueError(f"expected bbox with 4 or 5 values, got {box!r}")
    left = max(0, min(width, int(round(min(x1, x2)))))
    top = max(0, min(height, int(round(min(y1, y2)))))
    right = max(0, min(width, int(round(max(x1, x2)))))
    bottom = max(0, min(height, int(round(max(y1, y2)))))
    if right - left < min_crop_size or bottom - top < min_crop_size:
        return False
    crop = image.crop((left, top, right, bottom))
    crop.save(dest)
    return True


def extract_object_crops(
    image_root: str | Path,
    bboxes_json: str | Path,
    output_dir: str | Path,
    *,
    min_crop_size: int = 32,
) -> Path:
    """Extract object crops from scene images using a bbox JSON manifest."""
    root = Path(image_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bboxes = _load_bboxes(Path(bboxes_json))
    crop_index = 0
    for key, boxes in bboxes.items():
        image_path = _resolve_image_path(root, key)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            for box in boxes:
                dest = output / f"crop_{crop_index:06d}.png"
                if _save_crop(rgb, box, dest, min_crop_size=min_crop_size):
                    crop_index += 1
    if crop_index == 0:
        raise ValueError(f"no object crops extracted from {image_root} using {bboxes_json}")
    return output


def _resolve_crop_dir(
    image_root: Path,
    *,
    crop_dir: str | Path | None,
    bboxes_json: str | Path | None,
    scratch_root: Path,
    min_crop_size: int,
) -> Path:
    if crop_dir is not None:
        path = Path(crop_dir)
        if not path.is_dir() or not _iter_images(path):
            raise ValueError(f"crop directory has no images: {path}")
        return path
    if bboxes_json is None:
        raise ValueError(
            "SceneFID requires pre-extracted crop directories or a bboxes JSON manifest "
            "(object crop + FID protocol)."
        )
    return extract_object_crops(image_root, bboxes_json, scratch_root, min_crop_size=min_crop_size)


def compute_scene_fid(
    reference: str | Path,
    generated: str | Path,
    *,
    reference_crops: str | Path | None = None,
    generated_crops: str | Path | None = None,
    reference_bboxes_json: str | Path | None = None,
    generated_bboxes_json: str | Path | None = None,
    min_crop_size: int = 32,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> float:
    """Compute FID on object crops extracted from scene images."""
    ref_root = Path(reference)
    gen_root = Path(generated)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if reference_crops is None or generated_crops is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="worldfoundry-scene-fid-")
            scratch = Path(temp_dir.name)
            ref_crops = _resolve_crop_dir(
                ref_root,
                crop_dir=reference_crops,
                bboxes_json=reference_bboxes_json,
                scratch_root=scratch / "reference_crops",
                min_crop_size=min_crop_size,
            )
            gen_crops = _resolve_crop_dir(
                gen_root,
                crop_dir=generated_crops,
                bboxes_json=generated_bboxes_json or reference_bboxes_json,
                scratch_root=scratch / "generated_crops",
                min_crop_size=min_crop_size,
            )
        else:
            ref_crops = _resolve_crop_dir(
                ref_root,
                crop_dir=reference_crops,
                bboxes_json=None,
                scratch_root=Path(reference_crops),
                min_crop_size=min_crop_size,
            )
            gen_crops = _resolve_crop_dir(
                gen_root,
                crop_dir=generated_crops,
                bboxes_json=None,
                scratch_root=Path(generated_crops),
                min_crop_size=min_crop_size,
            )
        return compute_fid(
            ref_crops,
            gen_crops,
            batch_size=batch_size,
            cuda=cuda,
            feature_extractor=feature_extractor,
            **kwargs,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir.name, ignore_errors=True)


__all__ = ["compute_scene_fid", "extract_object_crops"]
