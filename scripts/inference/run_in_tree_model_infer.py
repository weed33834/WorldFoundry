#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" if (ROOT / "src" / "worldfoundry").is_dir() else ROOT
PACKAGE_ROOT = SOURCE_ROOT / "worldfoundry"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
CKPT_ROOT = Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", str(ROOT / "cache" / "checkpoints")))
MODEL = os.environ["IN_TREE_MODEL"]
MODEL_PATH = os.environ.get("IN_TREE_MODEL_PATH")
DEVICE = os.environ.get("IN_TREE_DEVICE", "cuda")
PROMPT = os.environ.get("IN_TREE_PROMPT", "a cinematic scene, high quality")
OUTPUT_DIR = Path(os.environ.get("IN_TREE_OUTPUT_DIR", ROOT / "tmp" / "infer_outputs" / "in_tree" / MODEL))
OUTPUT_PATH = Path(os.environ.get("IN_TREE_OUTPUT_PATH", OUTPUT_DIR / f"{MODEL}.mp4"))
TEST_CASES = PACKAGE_ROOT / "data" / "test_cases"
DEFAULT_IMAGE = TEST_CASES / "test_image_case1/ref_image.png"
DEFAULT_IMAGE_2 = TEST_CASES / "images/002.png"
DEFAULT_VIDEO = TEST_CASES / "longcat_video/motorcycle.mp4"
DEFAULT_IMAGE_DIR = TEST_CASES / "test_image_seq_case1"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}
ASTRA_CAM_TYPE_TO_DIRECTION = {
    "1": "forward",
    "2": "camera_l",
    "3": "camera_r",
    "4": "forward_left",
    "5": "forward_right",
    "6": "s_curve",
    "7": "left_right",
}
ASTRA_DIRECTION_ALIASES = {
    "forward": "forward",
    "move_forward": "forward",
    "rotate_left": "camera_l",
    "left": "camera_l",
    "camera_l": "camera_l",
    "rotate_right": "camera_r",
    "right": "camera_r",
    "camera_r": "camera_r",
    "forward_left": "forward_left",
    "move_forward_left": "forward_left",
    "forward_right": "forward_right",
    "move_forward_right": "forward_right",
    "s_curve": "s_curve",
    "s-shaped": "s_curve",
    "left_right": "left_right",
}


def _configure_cuda_local_rank() -> None:
    global DEVICE

    local_rank_value = os.environ.get("LOCAL_RANK")
    if local_rank_value in {None, ""}:
        return

    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    local_rank = int(local_rank_value)
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return
    local_device = local_rank % device_count
    torch.cuda.set_device(local_device)
    if os.environ.get("IN_TREE_DEVICE") in {None, "", "cuda"}:
        DEVICE = f"cuda:{local_device}"


_configure_cuda_local_rank()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in {None, ""} else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in {None, ""} else float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _distributed_rank() -> int:
    value = os.environ.get("RANK")
    if value not in {None, ""}:
        return int(value)
    return 0


def _is_main_process() -> bool:
    return _distributed_rank() == 0


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return None if value in {None, ""} else float(value)


def _ckpt(*parts: str) -> str:
    return str(CKPT_ROOT.joinpath(*parts))


def _first_existing(*paths: str) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    return paths[-1]


def _require_path(path: str | Path, label: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def _looks_like_remote_model_ref(path: str | Path) -> bool:
    text = str(path).strip()
    if not text or text.startswith(("/", "~", ".")):
        return False
    if "://" in text:
        return True
    parts = text.split("/")
    return len(parts) == 2 and all(part and part not in {".", ".."} for part in parts)


def _require_model_ref(path: str | Path, label: str) -> str:
    resolved = Path(path).expanduser()
    if resolved.exists():
        return str(resolved)
    if _looks_like_remote_model_ref(path):
        return str(path)
    raise FileNotFoundError(f"{label} not found: {resolved}")


def _input_image_path() -> str:
    path = os.environ.get("IN_TREE_INPUT_PATH")
    if path:
        return _require_path(path, "input image")
    if DEFAULT_IMAGE.exists():
        return str(DEFAULT_IMAGE)
    return _require_path(DEFAULT_IMAGE_2, "input image")


def _input_video_path() -> str:
    return _require_path(
        os.environ.get("IN_TREE_VIDEO_PATH") or os.environ.get("IN_TREE_INPUT_PATH") or str(DEFAULT_VIDEO),
        "input video",
    )


def _worldcam_demo_inputs() -> dict[str, str]:
    data_dir = Path(
        os.environ.get(
            "IN_TREE_WORLDCAM_DATA_DIR",
            str(TEST_CASES / "worldcam"),
        )
    )
    case_id = os.environ.get("IN_TREE_WORLDCAM_CASE", "0")
    video_path = (
        os.environ.get("IN_TREE_VIDEO_PATH")
        or os.environ.get("IN_TREE_INPUT_PATH")
        or str(data_dir / f"{case_id}.mp4")
    )
    intrinsics_path = os.environ.get("IN_TREE_INTRINSICS_PATH") or str(
        data_dir / f"{case_id}_intrinsics_palindrome.npy"
    )
    extrinsics_path = (
        os.environ.get("IN_TREE_EXTRINSICS_PATH")
        or os.environ.get("IN_TREE_POSES_PATH")
        or str(data_dir / f"{case_id}_poses_palindrome.npy")
    )
    prompt = PROMPT
    captions_path = data_dir / "captions.json"
    if "IN_TREE_PROMPT" not in os.environ and captions_path.is_file():
        try:
            captions = json.loads(captions_path.read_text())
            video_name = Path(video_path).name
            for item in captions:
                if item.get("media_path") == video_name and item.get("caption"):
                    prompt = str(item["caption"])
                    break
        except Exception:
            prompt = PROMPT
    return {
        "video_path": _require_path(video_path, "WorldCam conditioning video"),
        "intrinsics_path": _require_path(intrinsics_path, "WorldCam intrinsics"),
        "extrinsics_path": _require_path(extrinsics_path, "WorldCam extrinsics"),
        "prompt": prompt,
    }


def _input_media_path() -> str:
    path = os.environ.get("IN_TREE_VIDEO_PATH") or os.environ.get("IN_TREE_INPUT_PATH")
    if path:
        return _require_path(path, "input media")
    if DEFAULT_VIDEO.exists():
        return str(DEFAULT_VIDEO)
    return _input_image_path()


def _image_dir() -> str:
    path = os.environ.get("IN_TREE_IMAGE_DIR") or os.environ.get("IN_TREE_INPUT_PATH") or str(DEFAULT_IMAGE_DIR)
    return _require_path(path, "input image directory")


def _load_image(path: str | None = None):
    from PIL import Image

    return Image.open(path or _input_image_path()).convert("RGB")


def _load_neoverse_frames(num_frames: int, *, width: int, height: int, resize_mode: str, static_scene: bool):
    from worldfoundry.synthesis.visual_generation.neoverse.runtime_env import ensure_neoverse_runtime

    ensure_neoverse_runtime()
    from worldfoundry.base_models.diffusion_model.diffsynth.utils.neoverse_auxiliary import load_video

    return load_video(
        _input_media_path(),
        num_frames,
        resolution=(width, height),
        resize_mode=resize_mode,
        static_scene=static_scene,
    )


def _parse_size(default_width: int, default_height: int) -> tuple[int, int]:
    raw = os.environ.get("IN_TREE_SIZE")
    if not raw:
        return default_height, default_width
    normalized = raw.lower().replace("x", "*")
    if "*" not in normalized:
        raise ValueError(f"IN_TREE_SIZE must be WxH, got: {raw}")
    width, height = normalized.split("*", maxsplit=1)
    return int(height), int(width)


def _interactions(default: Sequence[str]) -> list[str]:
    raw = os.environ.get("IN_TREE_INTERACTIONS", "")
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _astra_direction() -> str:
    raw = os.environ.get("IN_TREE_ASTRA_CAM_TYPE") or os.environ.get("IN_TREE_INTERACTIONS") or "4"
    value = raw.strip()
    if not value:
        value = "4"
    if "," in value:
        raise ValueError("Astra expects one --cam-type value from 1..7, not a comma-separated action list.")
    if value in ASTRA_CAM_TYPE_TO_DIRECTION:
        return ASTRA_CAM_TYPE_TO_DIRECTION[value]
    alias = value.casefold().replace("-", "_").replace(" ", "_")
    if alias in ASTRA_DIRECTION_ALIASES:
        return ASTRA_DIRECTION_ALIASES[alias]
    valid = ", ".join(f"{key}:{direction}" for key, direction in ASTRA_CAM_TYPE_TO_DIRECTION.items())
    raise ValueError(f"Astra --cam-type must be one of {valid}; got {raw!r}.")


def _camera_trajectory() -> list[int | float]:
    raw = os.environ.get("IN_TREE_CAMERA_TRAJECTORY") or os.environ.get("IN_TREE_INTERACTIONS")
    if not raw:
        return [100, 100, 0, 0, 30]
    alias = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if alias in {"default", "forward"}:
        return [100, 100, 0, 0, 30]
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        values.append(int(number) if number.is_integer() else number)
    if len(values) != 5:
        raise ValueError("ReCamMaster camera trajectory must contain 5 comma-separated numbers.")
    return values


def _pose_sequence(count: int = 4) -> list[np.ndarray]:
    poses = []
    for index in range(count):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array([0.05 * index, 0.0, 0.0], dtype=np.float64)
        poses.append(pose)
    return poses


def _image_paths_from_dir(path: str | Path) -> list[str]:
    root = Path(path)
    if root.is_file():
        return [str(root)]
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(item for item in root.iterdir() if item.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"No image files found under {root}")
    return [str(item) for item in paths]


def _detect_non_real_result(value: Any) -> None:
    if isinstance(value, dict):
        status = str(value.get("status", "")).lower()
        backend_quality = str(value.get("backend_quality", "")).lower()
        artifact_kind = str(value.get("artifact_kind", "")).lower()
        if status in {"blocked", "planned", "prepared", "request_plan"}:
            raise RuntimeError(f"Model returned non-executed status: {status}")
        if status in {"failed", "failure", "error", "errored", "canceled", "cancelled"}:
            raise RuntimeError(f"Model returned failed status: {status}")
        if "blocked" in backend_quality or "plan" in backend_quality:
            raise RuntimeError(f"Model returned non-executed backend_quality: {backend_quality}")
        if artifact_kind == "blocked_plan":
            raise RuntimeError("Model returned a blocked-plan artifact.")
        for key in ("result", "metadata"):
            if key in value:
                _detect_non_real_result(value[key])


def _existing_artifact(value: Any) -> Path | None:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        return path if path.is_file() else None
    if not isinstance(value, dict):
        return None
    for key in ("artifact_path", "generated_video_path", "generated_image_path", "output_path", "video_path"):
        raw = value.get(key)
        if raw:
            path = Path(raw).expanduser()
            if path.is_file():
                return path
    return None


def _to_numpy_video(frames: Any) -> np.ndarray:
    import torch

    if hasattr(frames, "videos"):
        frames = frames.videos
    if isinstance(frames, dict):
        for key in ("video", "frames", "images"):
            if key in frames and frames[key] is not None:
                frames = frames[key]
                break

    if torch.is_tensor(frames):
        tensor = frames.detach().cpu()
        while tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 4:
            if tensor.shape[0] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
                tensor = tensor.permute(1, 2, 3, 0)
            elif tensor.shape[1] in {1, 3, 4}:
                tensor = tensor.permute(0, 2, 3, 1)
        array = tensor.numpy()
    elif isinstance(frames, np.ndarray):
        array = frames
    elif isinstance(frames, (list, tuple)):
        arrays = []
        for frame in frames:
            if torch.is_tensor(frame):
                frame = frame.detach().cpu()
                if frame.ndim == 3 and frame.shape[0] in {1, 3, 4}:
                    frame = frame.permute(1, 2, 0)
                frame = frame.numpy()
            elif hasattr(frame, "convert"):
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
            arrays.append(frame)
        array = np.stack(arrays, axis=0)
    else:
        raise TypeError(f"Unsupported video output type: {type(frames)!r}")

    while array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 3, 0))
    elif array.ndim == 4 and array.shape[1] in {1, 3, 4}:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim != 4:
        raise ValueError(f"Expected 4D video array, got shape {array.shape}")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if np.nanmin(array) < 0.0 and np.nanmax(array) <= 1.0:
                array = (array + 1.0) * 127.5
            elif np.nanmax(array) <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _save_video_result(result: Any, output_path: Path, fps: int) -> str | None:
    if not _is_main_process():
        return None
    _detect_non_real_result(result)
    existing = _existing_artifact(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        if existing.resolve() != output_path.resolve():
            shutil.copyfile(existing, output_path)
        _require_media_file(output_path)
        return str(output_path)
    frames = result.get("video") if isinstance(result, dict) and "video" in result else result
    video = _to_numpy_video(frames)
    imageio.mimsave(str(output_path), video, fps=fps)
    _require_media_file(output_path)
    return str(output_path)


def _image_output_path() -> Path:
    if OUTPUT_PATH.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return OUTPUT_PATH
    return OUTPUT_PATH.with_suffix(".png")


def _require_nonempty_file(path: str | Path) -> None:
    resolved = Path(path)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Expected output file was not written: {resolved}")


def _require_media_file(path: str | Path) -> None:
    resolved = Path(path)
    _require_nonempty_file(resolved)
    suffix = resolved.suffix.lower()
    if suffix in IMAGE_EXTS:
        from PIL import Image

        with Image.open(resolved) as image:
            image.verify()
        return
    if suffix in VIDEO_EXTS:
        reader = imageio.get_reader(str(resolved))
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()
        if np.asarray(frame).size == 0:
            raise ValueError(f"Expected at least one decodable frame in {resolved}")
        return
    raise ValueError(f"Unsupported output media suffix for {resolved}")


def _require_nonempty_dir(path: str | Path) -> None:
    resolved = Path(path)
    if not resolved.is_dir() or not any(resolved.rglob("*")):
        raise FileNotFoundError(f"Expected output directory was not populated: {resolved}")


def _file_snapshot(root: str | Path) -> dict[Path, tuple[int, int]]:
    resolved = Path(root)
    if not resolved.exists():
        return {}
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in resolved.rglob("*")
        if path.is_file()
    }


def _require_new_or_updated_artifact(saved: str | Path, before: Mapping[Path, tuple[int, int]]) -> None:
    path = Path(saved)
    if path.is_file():
        current = (path.stat().st_mtime_ns, path.stat().st_size)
        if before.get(path) == current:
            raise RuntimeError(f"Output file was not updated by this run: {path}")
        return
    if path.is_dir():
        after = _file_snapshot(path)
        changed = [item for item, stat in after.items() if before.get(item) != stat]
        if not changed:
            raise RuntimeError(f"Output directory has no new or updated files from this run: {path}")


def _runtime_video_kwargs(default_frames: int, default_steps: int, default_fps: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_frames": _env_int("IN_TREE_FRAMES", default_frames),
        "num_inference_steps": _env_int("IN_TREE_STEPS", default_steps),
        "seed": _env_int("IN_TREE_SEED", 42),
        "fps": _env_int("IN_TREE_FPS", default_fps),
    }
    if os.environ.get("IN_TREE_SIZE"):
        height, width = _parse_size(0, 0)
        kwargs["height"] = height
        kwargs["width"] = width
    if os.environ.get("IN_TREE_DTYPE"):
        kwargs["dtype"] = os.environ["IN_TREE_DTYPE"]
    if os.environ.get("IN_TREE_SCHEDULER"):
        kwargs["scheduler"] = os.environ["IN_TREE_SCHEDULER"]
    if os.environ.get("IN_TREE_GUIDANCE_SCALE"):
        kwargs["guidance_scale"] = _env_float("IN_TREE_GUIDANCE_SCALE", 6.0)
    if os.environ.get("IN_TREE_NEGATIVE_PROMPT"):
        kwargs["negative_prompt"] = os.environ["IN_TREE_NEGATIVE_PROMPT"]
    if os.environ.get("IN_TREE_MAX_SEQUENCE_LENGTH"):
        kwargs["max_sequence_length"] = _env_int("IN_TREE_MAX_SEQUENCE_LENGTH", 226)
    return kwargs


def _run_runtime_video(cls: type, *, images: Any = None, default_frames: int = 49, default_steps: int = 30, default_fps: int = 16) -> str:
    kwargs = _runtime_video_kwargs(default_frames, default_steps, default_fps)
    fps = int(kwargs.pop("fps"))
    pipeline = cls.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        lazy=True,
        required_components=kwargs,
    )
    result = pipeline(
        prompt=PROMPT,
        images=images,
        output_path=str(OUTPUT_PATH),
        fps=fps,
        return_dict=True,
    )
    return _save_video_result(result, OUTPUT_PATH, fps)


def run() -> str:
    fps = _env_int("IN_TREE_FPS", 16)
    frames = _env_int("IN_TREE_FRAMES", 49)
    steps = _env_int("IN_TREE_STEPS", 30)
    seed = _env_int("IN_TREE_SEED", 42)

    if MODEL == "cogvideox-2b-t2v":
        from worldfoundry.pipelines.cogvideox.pipeline_cogvideox_2b_t2v import CogVideoX2bT2VPipeline

        return _run_runtime_video(CogVideoX2bT2VPipeline, default_frames=frames, default_steps=steps, default_fps=fps)

    if MODEL in {"cogvideox-5b-t2v", "cogvideox-1.5-5b-t2v"}:
        from worldfoundry.pipelines.cogvideox.pipeline_cogvideox_5b_t2v import CogVideoX5bT2VPipeline

        return _run_runtime_video(CogVideoX5bT2VPipeline, default_frames=frames, default_steps=steps, default_fps=fps)

    if MODEL in {"cogvideox-5b-i2v", "cogvideox-1.5-5b-i2v"}:
        from worldfoundry.pipelines.cogvideox.pipeline_cogvideox_5b_i2v import CogVideoX5bI2VPipeline

        return _run_runtime_video(CogVideoX5bI2VPipeline, images=_input_image_path(), default_frames=frames, default_steps=steps, default_fps=fps)

    if MODEL == "dynamicrafter-512-i2v":
        from worldfoundry.pipelines.dynamicrafter.pipeline_dynamicrafter_512_i2v import DynamiCrafter512I2VPipeline

        height, width = _parse_size(512, 320)
        pipeline = DynamiCrafter512I2VPipeline.from_pretrained(
            model_path=MODEL_PATH,
            device=DEVICE,
            lazy=True,
            required_components={
                "height": height,
                "width": width,
                "video_length": frames,
                "ddim_steps": steps,
                "seed": seed,
            },
        )
        result = pipeline(prompt=PROMPT, images=_input_image_path(), output_path=str(OUTPUT_PATH), fps=fps, return_dict=True)
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "dynamicrafter-1024-i2v":
        from worldfoundry.pipelines.dynamicrafter.pipeline_dynamicrafter_1024_i2v import DynamiCrafter1024I2VPipeline

        height, width = _parse_size(1024, 576)
        pipeline = DynamiCrafter1024I2VPipeline.from_pretrained(
            model_path=MODEL_PATH,
            device=DEVICE,
            lazy=True,
            required_components={
                "height": height,
                "width": width,
                "video_length": frames,
                "ddim_steps": steps,
                "seed": seed,
            },
        )
        result = pipeline(prompt=PROMPT, images=_input_image_path(), output_path=str(OUTPUT_PATH), fps=fps, return_dict=True)
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "ltx-video-i2v":
        from worldfoundry.pipelines.ltx_video.pipeline_ltx_video_i2v import LTXVideoI2VPipeline

        height, width = _parse_size(768, 512)
        pipeline = LTXVideoI2VPipeline.from_pretrained(
            model_path=MODEL_PATH,
            device=DEVICE,
            lazy=True,
            required_components={
                "height": height,
                "width": width,
                "num_frames": frames,
                "num_inference_steps": steps,
                "frame_rate": fps,
                "seed": seed,
            },
        )
        result = pipeline(prompt=PROMPT, images=_input_image_path(), output_path=str(OUTPUT_PATH), fps=fps, return_dict=True)
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "motionctrl":
        from worldfoundry.pipelines.component_pipelines import MotionCtrlPipeline

        height, width = _parse_size(256, 256)
        motion_steps = _env_int("IN_TREE_STEPS", 50)
        model_options: dict[str, Any] = {
            "ckpt_path": MODEL_PATH,
            "model_id": "motionctrl",
        }
        for env_name, option_name in [
            ("IN_TREE_CONFIG_PATH", "config_path"),
            ("IN_TREE_COND_DIR", "cond_dir"),
            ("IN_TREE_ADAPTER_CKPT", "adapter_ckpt"),
        ]:
            value = os.environ.get(env_name)
            if value:
                model_options[option_name] = value

        raw_interactions = _interactions([])
        motion_interactions: list[str] = []
        motion_kwargs: dict[str, Any] = {
            "height": height,
            "width": width,
            "infer_steps": motion_steps,
            "seed": seed,
            "cfg_scale": _env_float("IN_TREE_GUIDANCE_SCALE", 7.5),
        }
        cond_tokens = {"both", "camera_motion", "object_motion"}
        condtype = os.environ.get("IN_TREE_CONDTYPE")
        if raw_interactions:
            first = raw_interactions[0].replace("-", "_")
            if len(raw_interactions) == 1 and first in cond_tokens:
                condtype = first
            else:
                motion_interactions = raw_interactions
        if condtype:
            motion_kwargs["condtype"] = condtype.replace("-", "_")
        elif not motion_interactions:
            motion_kwargs["condtype"] = "both"
        ddim_eta = os.environ.get("IN_TREE_DDIM_ETA")
        if ddim_eta:
            motion_kwargs["ddim_eta"] = float(ddim_eta)
        cond_t = os.environ.get("IN_TREE_COND_T")
        if cond_t:
            motion_kwargs["cond_T"] = int(cond_t)
        temporal_guidance = os.environ.get("IN_TREE_GUIDANCE_SCALE_TEMPORAL")
        if temporal_guidance:
            motion_kwargs["unconditional_guidance_scale_temporal"] = float(temporal_guidance)

        pipeline = MotionCtrlPipeline.from_pretrained(
            model_path=model_options,
            device=DEVICE,
            model_id="motionctrl",
        )
        result = pipeline(
            prompt=os.environ["IN_TREE_PROMPT"] if "IN_TREE_PROMPT" in os.environ else "",
            interactions=motion_interactions,
            output_path=str(OUTPUT_PATH),
            fps=fps,
            return_dict=True,
            **motion_kwargs,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "animatediff":
        from worldfoundry.pipelines.component_pipelines import AnimateDiffPipeline

        height, width = _parse_size(256, 256)
        model_options: dict[str, Any] = {
            "motion_module_path": MODEL_PATH,
            "model_id": "animatediff",
            "sd15_path": os.environ.get("IN_TREE_SD15_PATH", _ckpt("stable-diffusion-v1-5")),
        }
        for env_name, option_name in [
            ("IN_TREE_BASE_MODEL_PATH", "base_model_path"),
            ("IN_TREE_DREAMBOOTH_MODEL_PATH", "dreambooth_model_path"),
            ("IN_TREE_INFERENCE_CONFIG", "inference_config"),
            ("IN_TREE_HF_HUB_CACHE", "hf_hub_cache"),
        ]:
            value = os.environ.get(env_name)
            if value:
                model_options[option_name] = value
        if os.environ.get("IN_TREE_NEGATIVE_PROMPT"):
            model_options["negative_prompt"] = os.environ["IN_TREE_NEGATIVE_PROMPT"]

        pipeline = AnimateDiffPipeline.from_pretrained(
            model_path=model_options,
            device=DEVICE,
            model_id="animatediff",
        )
        result = pipeline(
            prompt=PROMPT,
            output_path=str(OUTPUT_PATH),
            fps=fps,
            return_dict=True,
            num_frames=frames,
            num_inference_steps=steps,
            seed=seed,
            height=height,
            width=width,
            guidance_scale=_env_float("IN_TREE_GUIDANCE_SCALE", 7.5),
            negative_prompt=os.environ.get("IN_TREE_NEGATIVE_PROMPT", ""),
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL in {"zeroscope-576w", "zeroscope-xl"}:
        from worldfoundry.pipelines.component_pipelines import ZeroScopePipeline

        height, width = _parse_size(576, 320)
        pipeline = ZeroScopePipeline.from_pretrained(
            model_path={"model_path": MODEL_PATH, "model_id": "zeroscope"},
            device=DEVICE,
        )
        result = pipeline(
            prompt=PROMPT,
            output_path=str(OUTPUT_PATH),
            fps=fps,
            return_dict=True,
            num_frames=frames,
            num_inference_steps=steps,
            height=height,
            width=width,
            seed=seed,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "open-magvit2":
        from worldfoundry.pipelines.component_pipelines import OpenMAGVIT2Pipeline

        output_path = _image_output_path()
        model_options: dict[str, Any] = {
            "checkpoint_path": MODEL_PATH,
            "model_id": "open-magvit2",
            "class_id": _env_int("IN_TREE_CLASS_ID", 207),
            "batch_size": _env_int("IN_TREE_BATCH_SIZE", 1),
        }
        if os.environ.get("IN_TREE_CONFIG_PATH"):
            model_options["config_path"] = os.environ["IN_TREE_CONFIG_PATH"]
        pipeline = OpenMAGVIT2Pipeline.from_pretrained(
            model_path=model_options,
            device=DEVICE,
            model_id="open-magvit2",
        )
        result = pipeline(
            prompt=PROMPT,
            interactions=[str(_env_int("IN_TREE_CLASS_ID", 207))],
            output_path=str(output_path),
            return_dict=True,
            class_id=_env_int("IN_TREE_CLASS_ID", 207),
            batch_size=_env_int("IN_TREE_BATCH_SIZE", 1),
            steps=None if os.environ.get("IN_TREE_STEPS") in {None, ""} else steps,
            cfg_scale=os.environ.get("IN_TREE_GUIDANCE_SCALE", "4.0,4.0"),
        )
        return _save_video_result(result, output_path, fps)

    if MODEL == "show-o":
        from worldfoundry.pipelines.component_pipelines import ShowOPipeline

        output_path = _image_output_path()
        raw_size = os.environ.get("IN_TREE_SIZE", "")
        if raw_size:
            height, width = _parse_size(512, 512)
            if height != width:
                raise ValueError("Show-O expects a square --size, e.g. 512*512.")
            resolution = height
        else:
            resolution = _env_int("IN_TREE_RESOLUTION", 512)
        model_options: dict[str, Any] = {
            "pretrained_model_path": MODEL_PATH,
            "model_id": "show-o",
            "vq_model_path": os.environ.get("IN_TREE_VQ_MODEL_PATH", _ckpt("magvitv2")),
            "llm_model_path": os.environ.get("IN_TREE_LLM_MODEL_PATH", _ckpt("phi-1_5")),
            "resolution": resolution,
            "batch_size": _env_int("IN_TREE_BATCH_SIZE", 1),
            "guidance_scale": _env_float("IN_TREE_GUIDANCE_SCALE", 3.0),
            "generation_timesteps": steps,
        }
        pipeline = ShowOPipeline.from_pretrained(
            model_path=model_options,
            device=DEVICE,
            model_id="show-o",
        )
        result = pipeline(
            prompt=PROMPT,
            interactions=["t2i"],
            output_path=str(output_path),
            return_dict=True,
            mode="t2i",
            batch_size=_env_int("IN_TREE_BATCH_SIZE", 1),
            guidance_scale=_env_float("IN_TREE_GUIDANCE_SCALE", 3.0),
            generation_timesteps=steps,
            resolution=resolution,
        )
        return _save_video_result(result, output_path, fps)

    if MODEL in {"sana-video-2b-480p", "sana-video-2b-720p", "longsana-video-2b-480p"}:
        if MODEL == "sana-video-2b-480p":
            from worldfoundry.pipelines.sana.pipeline_sana import SanaVideo2b480pPipeline as SanaCls
        elif MODEL == "sana-video-2b-720p":
            from worldfoundry.pipelines.sana.pipeline_sana import SanaVideo2b720pPipeline as SanaCls
        else:
            from worldfoundry.pipelines.sana.pipeline_sana import LongsanaVideo2b480pPipeline as SanaCls

        pipeline = SanaCls.from_pretrained(model_path=MODEL_PATH, device=DEVICE, model_id=MODEL)
        result = pipeline(
            prompt=PROMPT,
            output_path=str(OUTPUT_PATH),
            fps=fps,
            return_dict=True,
            num_frames=frames,
            num_inference_steps=steps,
            cfg_scale=_env_float("IN_TREE_GUIDANCE_SCALE", 1.0 if MODEL == "longsana-video-2b-480p" else 6.0),
            seed=seed,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL in {"cosmos-predict2.5-2b", "cosmos-predict2.5-14b"}:
        from worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2p5 import CosmosPredict2p5Pipeline

        height, width = _parse_size(1280, 704)
        pipeline = CosmosPredict2p5Pipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={
                "text_encoder_model_path": _ckpt("Cosmos-Reason1-7B"),
                "vae_model_path": _ckpt("Wan2.1-T2V-1.3B"),
            },
            mode=os.environ.get("IN_TREE_MODE", "img2world"),
            device=DEVICE,
        )
        result = pipeline(
            prompt=PROMPT,
            image_path=_input_image_path(),
            num_frames=frames,
            num_inference_steps=steps,
            fps=fps,
            height=height,
            width=width,
            seed=seed,
            output_type="pil",
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "infinite-world":
        from worldfoundry.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline

        pipeline = InfiniteWorldPipeline.from_pretrained(model_path=MODEL_PATH, device=DEVICE)
        result = pipeline(
            images=_load_image(),
            prompt=PROMPT,
            interactions=_interactions(["forward", "camera_left", "forward", "camera_right"]),
            num_chunks=_env_int("IN_TREE_NUM_CHUNKS", 1),
            num_frames=frames,
            seed=seed,
            return_dict=True,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "neoverse":
        from worldfoundry.pipelines.neoverse.pipeline_neoverse import NeoVersePipeline
        from worldfoundry.synthesis.visual_generation.neoverse.worldfoundry_runtime import DEFAULT_PROMPT

        height, width = _parse_size(560, 336)
        neoverse_frames = _env_int("IN_TREE_FRAMES", 81)
        neoverse_steps = _env_int("IN_TREE_STEPS", 4)
        neoverse_seed = _env_int("IN_TREE_SEED", 42)
        neoverse_fps = _env_int("IN_TREE_FPS", 16)
        resize_mode = os.environ.get("IN_TREE_RESIZE_MODE", "center_crop")
        static_scene = _env_bool("IN_TREE_STATIC_SCENE", False)
        disable_lora = _env_bool("IN_TREE_DISABLE_LORA", False)
        guidance_scale = _env_float("IN_TREE_GUIDANCE_SCALE", 5.0 if disable_lora else 1.0)
        trajectory_file = os.environ.get("IN_TREE_TRAJECTORY_FILE") or None
        interactions = _interactions([])
        trajectory = None if trajectory_file or interactions else os.environ.get("IN_TREE_TRAJECTORY", "tilt_up")
        save_root = str(OUTPUT_PATH.with_suffix("")) if _env_bool("IN_TREE_VIS_RENDERING", False) else None

        pipeline = NeoVersePipeline.from_pretrained(
            model_path=MODEL_PATH,
            device=DEVICE,
            required_components={
                "height": height,
                "width": width,
                "resize_mode": resize_mode,
                "disable_lora": disable_lora,
                "enable_vram_management": _env_bool("IN_TREE_LOW_VRAM", False),
                "num_inference_steps": neoverse_steps,
                "cfg_scale": guidance_scale,
            },
        )
        images = _load_neoverse_frames(
            neoverse_frames,
            width=width,
            height=height,
            resize_mode=resize_mode,
            static_scene=static_scene,
        )
        result = pipeline(
            images=images,
            prompt=os.environ.get("IN_TREE_PROMPT", DEFAULT_PROMPT),
            negative_prompt=os.environ.get("IN_TREE_NEGATIVE_PROMPT", ""),
            interactions=interactions or None,
            predefined_trajectory=trajectory,
            trajectory_file=trajectory_file,
            static_scene=static_scene,
            zoom_ratio=_env_float("IN_TREE_ZOOM_RATIO", 1.0),
            trajectory_mode=os.environ.get("IN_TREE_TRAJECTORY_MODE", "relative"),
            angle=_env_optional_float("IN_TREE_ANGLE"),
            distance=_env_optional_float("IN_TREE_DISTANCE"),
            orbit_radius=_env_optional_float("IN_TREE_ORBIT_RADIUS"),
            num_frames=neoverse_frames,
            alpha_threshold=_env_float("IN_TREE_ALPHA_THRESHOLD", 1.0),
            seed=neoverse_seed,
            cfg_scale=guidance_scale,
            num_inference_steps=neoverse_steps,
            save_root=save_root,
            return_dict=True,
        )
        return _save_video_result(result, OUTPUT_PATH, neoverse_fps)

    if MODEL == "worldcam":
        from worldfoundry.pipelines.worldcam.pipeline_worldcam import WorldCamPipeline

        demo_inputs = _worldcam_demo_inputs()
        height, width = _parse_size(832, 480)
        pipeline = WorldCamPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={"wan_model_path": _ckpt("Wan2.1-T2V-1.3B")},
            device=DEVICE,
        )
        result = pipeline(
            prompt=demo_inputs["prompt"],
            video_path=demo_inputs["video_path"],
            intrinsics_path=demo_inputs["intrinsics_path"],
            extrinsics_path=demo_inputs["extrinsics_path"],
            return_dict=True,
            num_inference_steps=steps,
            num_ar_steps=_env_int("IN_TREE_NUM_AR_STEPS", 50),
            conditioning_frames=_env_int("IN_TREE_CONDITIONING_FRAMES", 65),
            height=height,
            width=width,
            seed=seed,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "recammaster":
        from worldfoundry.pipelines.kling.pipeline_recammaster import ReCamMasterPipeline

        height, width = _parse_size(832, 480)
        pipeline = ReCamMasterPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={"wan_model_path": _ckpt("Wan2.1-T2V-1.3B")},
            device=DEVICE,
        )
        result = pipeline(
            camera_trajectory=_camera_trajectory(),
            video_path=_input_video_path(),
            prompt=PROMPT,
            num_frames=frames,
            max_num_frames=frames,
            size=(height, width),
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "astra":
        from worldfoundry.pipelines.kling.pipeline_astra import AstraPipeline

        frames_per_generation = min(frames, _env_int("IN_TREE_ASTRA_FRAMES_PER_GENERATION", 8))
        pipeline = AstraPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={"wan_model_path": _ckpt("Wan2.1-T2V-1.3B")},
            device=DEVICE,
            total_frames_to_generate=frames,
            frames_per_generation=frames_per_generation,
        )
        result = pipeline(
            image_path=_input_image_path(),
            interactions={
                "prompt": PROMPT,
                "direction": _astra_direction(),
            },
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "hunyuan-gamecraft":
        from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline

        height, width = _parse_size(1216, 704)
        pipeline = HunyuanGameCraftPipeline.from_pretrained(model_path=MODEL_PATH, device=DEVICE, cpu_offload=False, seed=seed)
        gamecraft_interactions = _interactions(["w", "s", "d", "a"])
        speed_values = os.environ.get("IN_TREE_INTERACTION_SPEEDS")
        if speed_values:
            gamecraft_speeds = [float(item.strip()) for item in speed_values.split(",") if item.strip()]
        else:
            gamecraft_speeds = [_env_float("IN_TREE_INTERACTION_SPEED", 0.2)] * len(gamecraft_interactions)
        if len(gamecraft_speeds) != len(gamecraft_interactions):
            raise ValueError("IN_TREE_INTERACTION_SPEEDS must match IN_TREE_INTERACTIONS length.")
        result = pipeline(
            images=_load_image(),
            prompt=PROMPT,
            interactions=gamecraft_interactions,
            interaction_speed=gamecraft_speeds,
            size=(height, width),
            num_frames=frames,
            infer_steps=steps,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "hunyuan-world-voyager":
        from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_world_voyager import HunyuanWorldVoyagerPipeline

        condition_dir = os.environ.get("IN_TREE_CONDITION_DIR")
        required_components = {"represent_model_path": _ckpt("moge-vitl")}
        if condition_dir:
            required_components["skip_representation_model"] = True
        pipeline = HunyuanWorldVoyagerPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components=required_components,
            device=DEVICE,
            represent_render_dir=str(OUTPUT_DIR / "represent_render"),
            save_representation_video=False,
        )
        pipeline.rendering_args.seed = seed
        pipeline.rendering_args.infer_steps = steps
        pipeline.rendering_args.flow_shift = _env_float("IN_TREE_FLOW_SHIFT", 7.0)
        pipeline.rendering_args.embedded_cfg_scale = _env_float("IN_TREE_EMBEDDED_CFG_SCALE", 6.0)
        pipeline.rendering_args.ulysses_degree = _env_int(
            "IN_TREE_ULYSSES_DEGREE",
            int(os.environ.get("WORLD_SIZE", "1")),
        )
        pipeline.rendering_args.ring_degree = _env_int("IN_TREE_RING_DEGREE", 1)
        result = pipeline(
            images=None if condition_dir else _load_image(),
            interactions=_interactions(["forward"]),
            prompt=PROMPT,
            num_frames=frames,
            condition_dir=condition_dir,
            output_save_path=str(OUTPUT_DIR / "final_render"),
            i2v_stability=_env_bool("IN_TREE_I2V_STABILITY", True),
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "hunyuan-worldplay":
        from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline

        os.environ.setdefault(
            "HUNYUANWORLDPLAY_TEXT_ENCODER_PATH",
            _first_existing(
                _ckpt("HunyuanVideo-1.5", "text_encoder", "llm"),
                _ckpt("Qwen2.5-VL-7B-Instruct"),
                _ckpt("Qwen", "Qwen2.5-VL-7B-Instruct"),
                _ckpt("hfd", "Qwen--Qwen2.5-VL-7B-Instruct"),
                _ckpt("kairos-sensenova", "Qwen", "Qwen2.5-VL-7B-Instruct-AWQ"),
            ),
        )
        pipeline = HunyuanWorldPlayPipeline.from_pretrained(
            model_path=os.environ.get("IN_TREE_ACTION_CKPT", MODEL_PATH),
            required_components={"video_model_path": _ckpt("HunyuanVideo-1.5")},
            mode=os.environ.get("IN_TREE_MODE", "480p_i2v"),
            enable_offloading=True,
            device=DEVICE,
        )
        result = pipeline(
            prompt=PROMPT,
            images=_load_image(),
            interactions=", ".join(_interactions(["w-31"])),
            num_frames=frames,
            num_inference_steps=steps,
            seed=seed,
            few_step=True,
            chunk_latent_frames=4,
            model_type="ar",
            transformer_resident_ar_rollout=_env_bool("IN_TREE_TRANSFORMER_RESIDENT_AR_ROLLOUT", True),
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL in {"hunyuanworld-mirror", "hunyuan-mirror"}:
        from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_mirror import HunyuanMirrorPipeline

        output_dir = OUTPUT_DIR
        pipeline = HunyuanMirrorPipeline.from_pretrained(
            model_path=MODEL_PATH,
            output_path=str(output_dir),
            device=DEVICE,
        )
        result = pipeline(
            image_path=_image_paths_from_dir(_image_dir()),
            output_path=str(output_dir),
            apply_sky_mask=False,
        )
        _detect_non_real_result(result)
        pipeline.save_results(result)
        _require_nonempty_dir(output_dir)
        return str(output_dir)

    if MODEL in {"hy-world-2.0", "hy-world-2p0"}:
        from worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0 import HYWorld2Pipeline

        output_dir = OUTPUT_DIR
        pipeline = HYWorld2Pipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={"task": os.environ.get("IN_TREE_TASK", "worldrecon")},
            device=DEVICE,
        )
        result = pipeline(input_path=_image_dir(), output_path=str(output_dir))
        _detect_non_real_result(result)
        _require_nonempty_dir(output_dir)
        return str(output_dir)

    if MODEL in {"lyra1", "lyra-1"}:
        from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline
        from worldfoundry.pipelines.lyra.lyra_utils import prepare_lyra1_checkpoint_root

        lyra1_checkpoint_root = prepare_lyra1_checkpoint_root(MODEL_PATH)
        lyra1_height, lyra1_width = _parse_size(1280, 704)
        lyra1_num_gpus = _env_int("IN_TREE_NUM_GPUS", 1)
        pipeline = Lyra1Pipeline.from_pretrained(
            model_path=lyra1_checkpoint_root,
            required_components={
                "checkpoint_dir": lyra1_checkpoint_root,
                "static_ckpt_path": str(Path(lyra1_checkpoint_root) / "Lyra" / "lyra_static.pt"),
                "default_mode": os.environ.get("IN_TREE_MODE", "static"),
            },
            device=DEVICE,
        )
        result = pipeline(
            images=_load_image(),
            prompt=PROMPT,
            interactions=_interactions(["forward", "camera_l", "forward", "right"]),
            mode=os.environ.get("IN_TREE_MODE", "static"),
            output_dir=str(OUTPUT_DIR),
            return_dict=True,
            execute=True,
            num_video_frames=_env_int("IN_TREE_FRAMES", 121),
            num_steps=_env_int("IN_TREE_STEPS", 35),
            fps=fps,
            height=lyra1_height,
            width=lyra1_width,
            seed=_env_int("IN_TREE_SEED", 42),
            guidance=_env_float("IN_TREE_GUIDANCE_SCALE", 1.0),
            num_gpus=lyra1_num_gpus,
            multi_trajectory=True,
            offload_diffusion_transformer=True,
            offload_tokenizer=True,
            offload_text_encoder_model=True,
            offload_prompt_upsampler=True,
            offload_guardrail_models=True,
            disable_prompt_encoder=True,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL in {"lyra2", "lyra-2"}:
        from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline

        lyra2_resolution = _parse_size(832, 480)
        lyra2_seed = _env_int("IN_TREE_SEED", 1)
        lyra2_steps = _env_int("IN_TREE_STEPS", 35)
        pipeline = Lyra2Pipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={
                "checkpoint_dir": str(Path(MODEL_PATH) / "checkpoints/model"),
                "negative_prompt_path": str(Path(MODEL_PATH) / "checkpoints/text_encoder/negative_prompt.pt"),
                "da3_model_path_custom": str(Path(MODEL_PATH) / "checkpoints/recon/model.pt"),
                "load_runtime": True,
            },
            device=DEVICE,
        )
        result = pipeline(
            images=_load_image(),
            prompt=PROMPT,
            interactions=_interactions(["forward", "camera_l", "forward", "right"]),
            fps=fps,
            resolution=lyra2_resolution,
            output_dir=str(OUTPUT_DIR),
            seed=lyra2_seed,
            guidance=_env_optional_float("IN_TREE_GUIDANCE_SCALE"),
            num_sampling_step=lyra2_steps,
            execute=True,
            return_dict=True,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "wow":
        from worldfoundry.pipelines.wow.pipeline_wow import WoWArgs, WoWPipeline

        args = WoWArgs(gpu=0, steps=steps, seed=seed, num_frames=frames)
        pipeline = WoWPipeline.from_pretrained(
            model_path=MODEL_PATH,
            synthesis_args=args,
            device=DEVICE,
        )
        result = pipeline(input_path=_input_image_path(), text_prompt=PROMPT, args=args)
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "worldfm":
        from worldfoundry.pipelines.worldfm.pipeline_worldfm import WorldFMPipeline

        input_image = _load_image()
        k_matrix = np.array(
            [
                [722.91626, 0.0, input_image.width / 2.0],
                [0.0, 722.91626, input_image.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pipeline = WorldFMPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={
                "vae_path": str(Path(MODEL_PATH) / "vae"),
                "checkpoint_filename": os.environ.get("IN_TREE_WORLDFM_CHECKPOINT", "worldfm_2-step.pth"),
                "moge_pretrained": _ckpt("moge-2-vitl-normal"),
            },
            device=DEVICE,
        )
        result = pipeline(
            images=input_image,
            K=k_matrix,
            interactions=_pose_sequence(3),
            scene_name="worldfm_demo",
            panorama_path=os.environ.get("IN_TREE_PANORAMA_PATH") or None,
            output_dir=str(OUTPUT_DIR),
            fps=fps,
            return_dict=True,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "inspatio-world":
        from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world import InspatioWorldPipeline

        trajectory_path = os.environ.get("IN_TREE_TRAJECTORY_FILE") or os.environ.get("IN_TREE_TRAJECTORY") or None
        required_components = {
            "wan_model_path": os.environ.get("IN_TREE_WAN_MODEL_PATH", _ckpt("Wan2.1-T2V-1.3B")),
            "da3_model_path": os.environ.get(
                "IN_TREE_DA3_MODEL_PATH",
                _first_existing(_ckpt("DA3-SMALL"), _ckpt("DA3-BASE"), _ckpt("DA3-LARGE")),
            ),
            "florence_model_path": os.environ.get("IN_TREE_FLORENCE_MODEL_PATH", _ckpt("Florence-2-large")),
        }
        if os.environ.get("IN_TREE_CONFIG_PATH"):
            required_components["config_path"] = os.environ["IN_TREE_CONFIG_PATH"]
        if os.environ.get("IN_TREE_TAE_CHECKPOINT_PATH"):
            required_components["tae_checkpoint_path"] = os.environ["IN_TREE_TAE_CHECKPOINT_PATH"]
        if trajectory_path:
            required_components["traj_txt_path"] = trajectory_path

        pipeline = InspatioWorldPipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components=required_components,
            device=DEVICE,
        )
        result = pipeline(
            videos=_input_video_path(),
            traj_txt_path=trajectory_path,
            prompt=PROMPT,
            output_dir=str(OUTPUT_DIR / "inspatio_world_runtime"),
            return_dict=True,
            fps=fps,
            step1_gpus=os.environ.get("IN_TREE_STEP1_GPUS", "0"),
            step2_gpus=os.environ.get("IN_TREE_STEP2_GPUS", "0"),
            step3_gpus=os.environ.get("IN_TREE_STEP3_GPUS"),
            step3_nproc=_env_int("IN_TREE_STEP3_NPROC", 1),
            master_port=_env_int("IN_TREE_MASTER_PORT", 29513),
            show_progress=_env_bool("IN_TREE_SHOW_PROGRESS", True),
            skip_step2=_env_bool("IN_TREE_SKIP_STEP2", False),
            skip_step3=_env_bool("IN_TREE_SKIP_STEP3", False),
            relative_to_source=_env_bool("IN_TREE_RELATIVE_TO_SOURCE", False),
            rotation_only=_env_bool("IN_TREE_ROTATION_ONLY", False),
            adaptive_frame=not _env_bool("IN_TREE_DISABLE_ADAPTIVE_FRAME", False),
            freeze_repeat=_env_int("IN_TREE_FREEZE_REPEAT", 0),
            freeze_frame=None if os.environ.get("IN_TREE_FREEZE_FRAME") in {None, ""} else _env_int("IN_TREE_FREEZE_FRAME", 0),
            use_tae=_env_bool("IN_TREE_USE_TAE", False),
            compile_dit=_env_bool("IN_TREE_COMPILE_DIT", False),
        )
        if _env_bool("IN_TREE_SKIP_STEP3", False) and isinstance(result, dict) and not result.get("generated_video_path"):
            preview_candidates = sorted(
                Path(result["input_dir"]).glob("new_vggt/*/render/render_offline.mp4")
            )
            if preview_candidates:
                return _save_video_result({"generated_video_path": str(preview_candidates[0])}, OUTPUT_PATH, fps)
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL == "longvie2":
        from worldfoundry.pipelines.longvie.pipeline_longvie import LongVie2Pipeline

        dense_video = os.environ.get("IN_TREE_DENSE_VIDEO_PATH") or os.environ.get("IN_TREE_VIDEO_PATH") or str(DEFAULT_VIDEO)
        sparse_video = os.environ.get("IN_TREE_SPARSE_VIDEO_PATH") or dense_video
        pipeline = LongVie2Pipeline.from_pretrained(
            model_path=MODEL_PATH,
            required_components={
                "wan_base_dir": _ckpt("Wan2.1-I2V-14B-480P"),
                "tokenizer_dir": _ckpt("Wan2.1-T2V-1.3B"),
            },
            device=DEVICE,
            model_id="longvie-2",
        )
        result = pipeline(
            prompt=PROMPT,
            images=_input_image_path(),
            video={
                "dense_video": _require_path(dense_video, "LongVie dense video"),
                "sparse_video": _require_path(sparse_video, "LongVie sparse video"),
            },
            output_path=str(OUTPUT_PATH),
            fps=fps,
            return_dict=True,
            execute=True,
            seed=seed,
            num_frames=frames,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    if MODEL in {"fantasy-world-wan21", "fantasy-world-wan22"}:
        if MODEL == "fantasy-world-wan21":
            from worldfoundry.pipelines.fantasy_world.pipeline_fantasy_world import FantasyWorldWan21Pipeline as FantasyCls

            required_components = {
                "wan_model_path": _ckpt("Wan2.1-I2V-14B-480P"),
                "moge_pretrained": _ckpt("moge-2-vitl-normal"),
                "frames": frames,
                "sample_steps": steps,
                "fps": fps,
            }
        else:
            from worldfoundry.pipelines.fantasy_world.pipeline_fantasy_world import FantasyWorldWan22Pipeline as FantasyCls

            required_components = {
                "wan_model_path": _first_existing(
                    _ckpt("Wan2.2-I2V-A14B"),
                    _ckpt("Wan2.2-TI2V-5B"),
                ),
                "lora_path": _first_existing(_ckpt("Wan2.2-Fun-Reward-LoRAs"), MODEL_PATH),
                "moge_pretrained": _ckpt("moge-2-vitl-normal"),
                "frames": frames,
                "sample_steps": steps,
                "fps": fps,
            }
        pipeline = FantasyCls.from_pretrained(model_path=MODEL_PATH, required_components=required_components, device=DEVICE)
        result = pipeline(
            images=_load_image(),
            interactions=_pose_sequence(4),
            prompt=PROMPT,
            output_dir=str(OUTPUT_DIR),
            return_dict=True,
        )
        return _save_video_result(result, OUTPUT_PATH, fps)

    raise KeyError(f"Unsupported IN_TREE_MODEL: {MODEL}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH is not None:
        _require_model_ref(MODEL_PATH, "model path")
    before = _file_snapshot(OUTPUT_DIR)
    saved = run()
    if not _is_main_process() and saved is None:
        payload = {"model": MODEL, "rank": _distributed_rank(), "saved": None}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    _require_new_or_updated_artifact(saved, before)
    payload = {"model": MODEL, "saved": saved}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
