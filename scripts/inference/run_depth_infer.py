#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT if (REPO_ROOT / "worldfoundry").is_dir() else REPO_ROOT
PACKAGE_ROOT = SOURCE_ROOT / "worldfoundry"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
DEFAULT_INPUT = str(PACKAGE_ROOT / "data" / "test_cases" / "test_image_case1" / "ref_image.png")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _device() -> str:
    return os.environ.get("DEPTH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _input_path() -> str:
    return os.environ.get("DEPTH_INPUT_PATH") or os.environ.get("DEPTH_DATA_PATH") or DEFAULT_INPUT


def _output_dir() -> Path:
    output_dir = Path(os.environ.get("DEPTH_OUTPUT_DIR", "./depth_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_rgb_tensor(image_path: str, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def _normalized_depth_image(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(depth)
    valid = finite if mask is None else (finite & mask.astype(bool))
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid_depth = depth[valid]
    depth_min = float(valid_depth.min())
    depth_max = float(valid_depth.max())
    normalized = np.zeros_like(depth, dtype=np.float32)
    if depth_max > depth_min:
        normalized[valid] = (depth[valid] - depth_min) / (depth_max - depth_min)
    depth_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    color[~valid] = 0
    return color


def _save_depth_arrays(
    output_dir: Path,
    stem: str,
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    saved: list[str] = []
    depth_path = output_dir / f"{stem}_depth.npy"
    np.save(depth_path, depth)
    saved.append(str(depth_path))

    vis_path = output_dir / f"{stem}_depth.png"
    cv2.imwrite(str(vis_path), _normalized_depth_image(depth, mask=mask))
    saved.append(str(vis_path))

    if mask is not None:
        mask_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
        saved.append(str(mask_path))

    if extra:
        for key, value in extra.items():
            value_path = output_dir / f"{stem}_{key}.npy"
            np.save(value_path, value)
            saved.append(str(value_path))
    return saved


def _optional_batch_item(value: Any, index: int, batch_size: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return None
    if array.shape[0] == batch_size:
        return array[index]
    if batch_size == 1:
        return array
    return None


def _optional_visualization(value: Any, index: int, batch_size: int) -> np.ndarray | None:
    array = _optional_batch_item(value, index, batch_size)
    if array is None or array.ndim < 2:
        return None
    return array


def _run_depth_anything_v1() -> list[str]:
    from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v1 import (
        DepthAnything1Pipeline,
    )

    model_path = os.environ.get("DEPTH_MODEL_PATH")
    encoder = os.environ.get("DEPTH_ENCODER", "vitl")
    output_dir = _output_dir()
    pipeline = DepthAnything1Pipeline.from_pretrained(
        pretrained_model_path=model_path,
        encoder=encoder,
        device=_device(),
        data_type=os.environ.get("DEPTH_DATA_TYPE", "image"),
    )
    result = pipeline(
        _input_path(),
        grayscale=_bool_env("DEPTH_GRAYSCALE", False),
    )
    return result.save(str(output_dir))


def _run_depth_anything_v2() -> list[str]:
    from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v2 import (
        DepthAnything2Pipeline,
    )

    output_dir = _output_dir()
    pipeline = DepthAnything2Pipeline.from_pretrained(
        pretrained_model_path=os.environ.get("DEPTH_MODEL_PATH"),
        encoder=os.environ.get("DEPTH_ENCODER", "vitl"),
        device=_device(),
        data_type=os.environ.get("DEPTH_DATA_TYPE", "image"),
        default_input_size=int(os.environ.get("DEPTH_INPUT_SIZE", "518")),
    )
    result = pipeline(
        _input_path(),
        grayscale=_bool_env("DEPTH_GRAYSCALE", False),
        input_size=int(os.environ.get("DEPTH_INPUT_SIZE", "518")),
    )
    return result.save(str(output_dir))


def _run_depth_anything_v3() -> list[str]:
    from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v3 import (
        DepthAnything3Pipeline,
    )

    output_dir = _output_dir()
    pipeline = DepthAnything3Pipeline.from_pretrained(
        pretrained_model_path=os.environ.get("DEPTH_MODEL_PATH"),
        device=_device(),
    )
    result = pipeline(
        input_data=_input_path(),
        output_dir=str(output_dir),
        export_format=os.environ.get("DEPTH_EXPORT_FORMAT", "depth_vis"),
        process_res=int(os.environ.get("DEPTH_PROCESS_RES", "504")),
    )

    saved: list[str] = []
    depth = np.asarray(result["depth"])
    visualizations = np.asarray(result["depth_visualizations"])
    if depth.ndim == 2:
        depth = depth[None, ...]
    batch_size = depth.shape[0]
    input_stem = Path(_input_path()).stem
    for index in range(batch_size):
        suffix = input_stem if batch_size == 1 else f"{input_stem}_{index:04d}"
        extra = {
            key: item
            for key in ("confidence", "sky")
            for item in (_optional_batch_item(result.get(key), index, batch_size),)
            if item is not None
        }
        saved.extend(
            _save_depth_arrays(
                output_dir,
                suffix,
                depth[index],
                extra=extra,
            )
        )
        visualization = _optional_visualization(visualizations, index, batch_size)
        if visualization is not None:
            vis_path = output_dir / f"{suffix}_da3_depth.png"
            cv2.imwrite(str(vis_path), visualization)
            saved.append(str(vis_path))
    return saved


def _run_moge() -> list[str]:
    from worldfoundry.base_models.three_dimensions.depth.moge.model import (
        import_model_class_by_version,
    )

    model_path = _resolve_moge_model_path(os.environ["DEPTH_MODEL_PATH"])
    version = os.environ.get("DEPTH_MOGE_VERSION", "v2")
    device = _device()
    model_cls = import_model_class_by_version(version)
    model = model_cls.from_pretrained(model_path).to(device).eval()

    image = _load_rgb_tensor(_input_path(), device=device)
    use_fp16 = _bool_env("DEPTH_MOGE_USE_FP16", True) and device.startswith("cuda")
    output = model.infer(
        image,
        resolution_level=int(os.environ.get("DEPTH_MOGE_RESOLUTION_LEVEL", "9")),
        apply_mask=_bool_env("DEPTH_MOGE_APPLY_MASK", True),
        force_projection=_bool_env("DEPTH_MOGE_FORCE_PROJECTION", True),
        use_fp16=use_fp16,
    )

    depth = output["depth"].detach().cpu().float().numpy()
    mask_tensor = output.get("mask")
    mask = None if mask_tensor is None else mask_tensor.detach().cpu().numpy().astype(bool)
    extra = {
        "intrinsics": output["intrinsics"].detach().cpu().float().numpy(),
    }
    if "normal" in output:
        extra["normal"] = output["normal"].detach().cpu().float().numpy()

    return _save_depth_arrays(_output_dir(), Path(_input_path()).stem, depth, mask=mask, extra=extra)


def _resolve_moge_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.exists():
        return value
    if path.is_file():
        return str(path)
    checkpoint_path = path / "model.pt"
    if checkpoint_path.is_file():
        return str(checkpoint_path)
    raise FileNotFoundError(f"Expected MoGE checkpoint directory to contain model.pt: {path}")


def main() -> None:
    family = os.environ["DEPTH_MODEL_FAMILY"]
    runners = {
        "depth_anything_v1": _run_depth_anything_v1,
        "depth_anything_v2": _run_depth_anything_v2,
        "depth_anything_v3": _run_depth_anything_v3,
        "moge": _run_moge,
    }
    if family not in runners:
        raise ValueError(f"Unsupported DEPTH_MODEL_FAMILY={family!r}")

    saved = runners[family]()
    print("Saved files:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
