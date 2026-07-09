import os
import sys
from pathlib import Path
from typing import Any, Iterable

import imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" if (ROOT / "src" / "worldfoundry").is_dir() else ROOT
PACKAGE_ROOT = SOURCE_ROOT / "worldfoundry"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DEFAULT_IMAGE = str(PACKAGE_ROOT / "data" / "test_cases" / "test_image_case1" / "ref_image.png")
DEFAULT_PROMPT = "A cozy medieval-style village square on a winter evening, with timber-framed cottages."


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _split_csv(value: str | None, default: Iterable[str]) -> list[str]:
    raw = value if value is not None else ",".join(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _output_formats() -> tuple[bool, bool, bool]:
    raw = os.getenv("THREED_OUTPUT_FORMATS", "video,spz,ply")
    formats = {item.strip().lower() for item in raw.split(",") if item.strip()}
    save_video = _env_bool("THREED_SAVE_VIDEO", "video" in formats or not formats)
    save_spz = _env_bool("THREED_SAVE_SPZ", "spz" in formats or not formats)
    save_ply = _env_bool("THREED_SAVE_PLY", "ply" in formats or not formats)
    return save_video, save_spz, save_ply


def _json_paths_from_dir(path_value: str) -> list[Path]:
    root = Path(path_value).expanduser()
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"FlashWorld input_dir not found: {root}")
    paths = sorted(item for item in root.iterdir() if item.suffix.lower() == ".json")
    if not paths:
        raise FileNotFoundError(f"No FlashWorld JSON files found under: {root}")
    return paths


def _require_existing_path(path_value: str, label: str) -> str:
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path)
    if path.is_absolute() or path_value.startswith("."):
        raise FileNotFoundError(f"{label} not found: {path_value}")
    return path_value


def _frame_to_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255)
        else:
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _save_video_frames(frames: list[Any], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [_frame_to_array(frame) for frame in frames], fps=fps)


def _save_image(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, Image.Image):
        frame.convert("RGB").save(path)
    else:
        Image.fromarray(_frame_to_array(frame)).save(path)


def _run_pi3_like(family: str) -> None:
    if family in {"pi3", "pi3x"}:
        from worldfoundry.pipelines.pi3.pipeline_pi3 import Pi3Pipeline

        pipeline_cls = Pi3Pipeline
        mode = os.getenv("THREED_MODEL_MODE", family)
    else:
        from worldfoundry.pipelines.pi3.pipeline_loger import LoGeRPipeline

        pipeline_cls = LoGeRPipeline
        mode = os.getenv("THREED_MODEL_MODE", "loger_star" if family == "loger_star" else "loger")

    model_path = _require_existing_path(os.environ["THREED_MODEL_PATH"], "3D model path")
    output_dir = Path(os.getenv("THREED_OUTPUT_DIR", f"./output_{family}")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = os.getenv("THREED_INPUT_PATH", DEFAULT_IMAGE)
    input_video = os.getenv("THREED_VIDEO_PATH")
    visual_input = _require_existing_path(input_video or input_image, "3D input")
    interactions = _split_csv(
        os.getenv("THREED_INTERACTIONS"),
        ["forward", "left", "camera_r"],
    )
    fps = _env_int("THREED_FPS", 15)
    task_type = os.getenv("THREED_TASK_TYPE", "all").strip().lower()
    interval = _env_int("THREED_INTERVAL", 10)

    pipeline = pipeline_cls.from_pretrained(model_path=model_path, mode=mode)
    if input_video:
        result = pipeline(videos=visual_input, task_type="reconstruction", interval=interval)
    else:
        result = pipeline(images=visual_input, task_type="reconstruction", interval=interval)

    saved_files = result.save(str(output_dir))

    if task_type in {"all", "render_view"}:
        rendered = pipeline(task_type="render_view", camera_view=0)
        _save_image(rendered, output_dir / "render_default.png")
        frames = pipeline(task_type="render_view", interactions=interactions)
        _save_video_frames(frames, output_dir / "interaction_video.mp4", fps=fps)

    if task_type in {"all", "render_trajectory"}:
        frames = pipeline(task_type="render_trajectory")
        _save_video_frames(frames, output_dir / "trajectory_video.mp4", fps=fps)

    print(f"{family} outputs saved to: {output_dir}")
    print(f"Saved reconstruction files: {len(saved_files)}")


def _run_flash_world() -> None:
    from worldfoundry.pipelines.flash_world.pipeline_flash_world import FlashWorldPipeline

    model_path = _require_existing_path(os.environ["THREED_MODEL_PATH"], "FlashWorld model path")
    wan_model_path = _require_existing_path(os.environ["THREED_WAN_MODEL_PATH"], "FlashWorld Wan model path")
    output_dir = Path(os.getenv("THREED_OUTPUT_DIR", "./output_flash_world")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = _env_int("THREED_FPS", 15)
    height = _env_int("THREED_IMAGE_HEIGHT", 480)
    width = _env_int("THREED_IMAGE_WIDTH", 704)
    num_frames = _env_int("THREED_NUM_FRAMES", 24)
    interactions = _split_csv(
        os.getenv("THREED_INTERACTIONS"),
        ["forward", "camera_l", "right", "camera_zoom_in"],
    )
    save_video, save_spz, save_ply = _output_formats()

    pipeline = FlashWorldPipeline.from_pretrained(
        model_path=model_path,
        required_components={"wan_model_path": wan_model_path},
        offload_t5=_env_bool("THREED_OFFLOAD_T5", False),
        offload_vae=_env_bool("THREED_OFFLOAD_VAE", False),
        offload_transformer_during_vae=_env_bool("THREED_OFFLOAD_TRANSFORMER_DURING_VAE", False),
    )

    input_dir = os.getenv("THREED_INPUT_DIR")
    if input_dir:
        for json_path in _json_paths_from_dir(input_dir):
            item_output_dir = output_dir / json_path.stem
            config = pipeline.load_config_from_json(
                str(json_path),
                output_dir=str(item_output_dir),
                default_config={
                    "text_prompt": "",
                    "image_prompt": None,
                    "resolution": [num_frames, height, width],
                    "image_index": 0,
                    "cameras": None,
                    "return_video": save_video,
                    "video_fps": fps,
                },
            )
            results = pipeline(
                images=config["input_"],
                prompt=config["text_prompt"],
                camera_view=config["cameras"],
                num_frames=config["num_frames"],
                fps=fps,
                image_height=config["image_height"],
                image_width=config["image_width"],
                image_index=config["image_index"],
                return_video=save_video,
            )
            saved = pipeline.save_results(
                results=results,
                output_dir=str(item_output_dir),
                save_ply=save_ply,
                save_spz=save_spz,
                save_video=save_video,
            )
            print(f"FlashWorld JSON {json_path.name} outputs saved to: {item_output_dir}")
            for key, value in sorted(saved.items()):
                print(f"{json_path.stem}.{key}: {value}")
        return

    input_path = _require_existing_path(os.getenv("THREED_INPUT_PATH", DEFAULT_IMAGE), "FlashWorld input image")
    results = pipeline(
        images=input_path,
        prompt=os.getenv("THREED_PROMPT", DEFAULT_PROMPT),
        interactions=interactions,
        num_frames=num_frames,
        fps=fps,
        image_height=height,
        image_width=width,
        image_index=_env_int("THREED_IMAGE_INDEX", 0),
        return_video=save_video,
    )
    saved = pipeline.save_results(
        results=results,
        output_dir=str(output_dir),
        save_ply=save_ply,
        save_spz=save_spz,
        save_video=save_video,
    )
    print(f"FlashWorld outputs saved to: {output_dir}")
    for key, value in sorted(saved.items()):
        print(f"{key}: {value}")


family = os.environ.get("THREED_MODEL_FAMILY", "").strip().lower()
if family in {"pi3", "pi3x", "loger", "loger_star"}:
    _run_pi3_like(family)
elif family in {"flash_world", "flashworld"}:
    _run_flash_world()
else:
    raise ValueError(
        "THREED_MODEL_FAMILY must be one of pi3, pi3x, loger, loger_star, flash_world."
    )
