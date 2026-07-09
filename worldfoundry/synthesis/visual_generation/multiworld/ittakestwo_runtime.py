from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from worldfoundry.core import load_pil_image, load_video_frames, save_video_frames
from worldfoundry.core.io.paths import package_module_root as package_root

from .runtime_env import (
    WAN_TI2V_DIT_FILENAMES,
    build_subprocess_env,
    resolve_checkpoint_path,
    resolve_config_path,
    resolve_runtime_root,
    resolve_wan_ti2v_root,
)


DIFFSYNTH_PARENT = package_root("worldfoundry.base_models.diffusion_model.diffsynth").parent


def save_image_input(image_input: Any, output_path: str | Path) -> str:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_pil_image(image_input).save(output_path)
    return str(output_path)


def _numeric_array_from_sequence(value: Sequence[Any]) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0 or array.dtype == object:
        return None
    if array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None
    return array


def _to_numpy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_numpy_tree(child) for key, child in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        array = _numeric_array_from_sequence(value)
        if array is not None:
            return array
        return [_to_numpy_tree(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        return value
    return np.asarray(value)


def dump_tree(value: Any, output_path: str | Path) -> str:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(_to_numpy_tree(value), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(output_path)


def load_tree(input_path: str | Path) -> Any:
    with Path(input_path).expanduser().resolve().open("rb") as handle:
        return pickle.load(handle)


def load_ittakestwo_action_csv(
    action_path: str | Path,
    *,
    num_frames: int = 81,
    stick_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Load an official ItTakesTwo action CSV into the 10/2-dim dual-player action format."""

    resolved = Path(action_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MultiWorld ItTakesTwo action CSV not found: {resolved}")

    def as_int(row: Dict[str, str], key: str) -> int:
        try:
            return int(float(row.get(key) or 0))
        except ValueError:
            return 0

    def as_float(row: Dict[str, str], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except ValueError:
            return 0.0

    discrete: list[list[list[int]]] = []
    continuous: list[list[list[float]]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left_discrete = [as_int(row, key) for key in ("w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "f")]
            left_continuous = [as_float(row, "norm_dx"), as_float(row, "norm_dy")]

            right_discrete = [0] * 10
            right_continuous = [as_float(row, "axis_2"), as_float(row, "axis_3")]
            if as_int(row, "button_0"):
                right_discrete[4] = 1
            if as_int(row, "button_1"):
                right_discrete[6] = 1
            if as_int(row, "button_2") or as_int(row, "button_8"):
                right_discrete[5] = 1
            if as_int(row, "button_3"):
                right_discrete[9] = 1
            if as_int(row, "button_4"):
                right_discrete[8] = 1
            if as_int(row, "button_5"):
                right_discrete[7] = 1

            axis_0 = as_float(row, "axis_0")
            axis_1 = as_float(row, "axis_1")
            if axis_0 < -stick_threshold:
                right_discrete[1] = 1
            elif axis_0 > stick_threshold:
                right_discrete[3] = 1
            if axis_1 < -stick_threshold:
                right_discrete[0] = 1
            elif axis_1 > stick_threshold:
                right_discrete[2] = 1

            discrete.append([left_discrete, right_discrete])
            continuous.append([left_continuous, right_continuous])
            if len(discrete) >= num_frames:
                break

    if len(discrete) < num_frames:
        raise ValueError(
            f"MultiWorld ItTakesTwo action CSV has {len(discrete)} frames, "
            f"but {num_frames} frames were requested: {resolved}"
        )
    return {
        "discrete_action": [discrete],
        "continuous_action": [continuous],
    }


class MultiWorldItTakesTwoRuntime:
    """WorldFoundry adapter for the official MultiWorld ItTakesTwo runtime."""

    def __init__(
        self,
        runtime_root: str,
        config_path: str,
        checkpoint_path: str,
        *,
        python_executable: Optional[str] = None,
        device: str = "cuda",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.runtime_root = str(Path(runtime_root).expanduser().resolve())
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.python_executable = python_executable or sys.executable
        self.device = device
        self.defaults = {
            "derive_env_obv_from_image": True,
            "num_inference_steps": 35,
            "inference_seed": 0,
            "fps": None,
        }
        if defaults:
            self.defaults.update(defaults)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        args=None,
        device: Optional[str] = None,
        runtime_root: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        python_executable: Optional[str] = None,
        derive_env_obv_from_image: bool = True,
        num_inference_steps: int = 35,
        inference_seed: int = 0,
        fps: Optional[int] = None,
        **kwargs,
    ) -> "MultiWorldItTakesTwoRuntime":
        del args
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported MultiWorld ItTakesTwo kwargs: {unknown}")

        resolved_runtime_root = resolve_runtime_root(runtime_root)
        resolved_config_path = resolve_config_path(config_path, resolved_runtime_root)
        resolved_checkpoint_path = resolve_checkpoint_path(
            checkpoint_path or pretrained_model_path,
            resolved_runtime_root,
        )
        defaults = {
            "derive_env_obv_from_image": bool(derive_env_obv_from_image),
            "num_inference_steps": int(num_inference_steps),
            "inference_seed": int(inference_seed),
            "fps": None if fps is None else int(fps),
        }
        return cls(
            runtime_root=resolved_runtime_root,
            config_path=resolved_config_path,
            checkpoint_path=resolved_checkpoint_path,
            python_executable=python_executable,
            device=device or "cuda",
            defaults=defaults,
        )

    def predict(
        self,
        image,
        action: Dict[str, Any],
        env_obv: Any = None,
        output_dir: Optional[str] = None,
        save_name: str = "multiworld_ittakestwo",
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        inference_seed: Optional[int] = None,
        derive_env_obv_from_image: Optional[bool] = None,
        return_dict: bool = False,
        show_progress: bool = True,
    ):
        if not isinstance(action, dict):
            raise TypeError("MultiWorld ItTakesTwo expects action to be a dict of arrays or tensors.")

        output_root = Path(output_dir or tempfile.mkdtemp(prefix="multiworld_ittakestwo_"))
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        image_path = save_image_input(image, output_root / "input.png")
        action_path = dump_tree(action, output_root / "action.pkl")
        env_obv_path = None if env_obv is None else dump_tree(env_obv, output_root / "env_obv.pkl")

        command = [
            self.python_executable,
            "-m",
            "worldfoundry.synthesis.visual_generation.multiworld.ittakestwo_runtime",
            "--runtime_root",
            self.runtime_root,
            "--config_path",
            self.config_path,
            "--checkpoint_path",
            self.checkpoint_path,
            "--input_image_path",
            image_path,
            "--action_path",
            action_path,
            "--output_dir",
            str(output_root),
            "--save_name",
            save_name,
            "--device",
            self.device,
            "--num_inference_steps",
            str(
                int(
                    num_inference_steps
                    if num_inference_steps is not None
                    else self.defaults["num_inference_steps"]
                )
            ),
            "--inference_seed",
            str(
                int(
                    inference_seed
                    if inference_seed is not None
                    else self.defaults["inference_seed"]
                )
            ),
        ]
        if env_obv_path is not None:
            command.extend(["--env_obv_path", env_obv_path])
        elif (
            self.defaults["derive_env_obv_from_image"]
            if derive_env_obv_from_image is None
            else bool(derive_env_obv_from_image)
        ):
            command.append("--derive_env_obv_from_image")
        if num_frames is not None:
            command.extend(["--num_frames", str(int(num_frames))])
        if height is not None:
            command.extend(["--height", str(int(height))])
        if width is not None:
            command.extend(["--width", str(int(width))])
        resolved_fps = fps if fps is not None else self.defaults["fps"]
        if resolved_fps is not None:
            command.extend(["--fps", str(int(resolved_fps))])

        env = build_subprocess_env(self.runtime_root)
        try:
            completed = subprocess.run(
                command,
                check=True,
                cwd=self.runtime_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            output = (error.stdout or "").strip()
            if output:
                print(output)
            tail = output[-4000:] if output else "no subprocess output captured"
            raise RuntimeError(
                "MultiWorld ItTakesTwo generation failed "
                f"(exit code {error.returncode}). Subprocess output tail:\n{tail}"
            ) from error
        if show_progress and completed.stdout:
            print(completed.stdout)

        metadata_path = output_root / f"{save_name}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"MultiWorld metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        video_path = Path(metadata["generated_video_path"]).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"MultiWorld output video not found: {video_path}")

        video = load_video_frames(video_path)
        result = {
            "video": video,
            "frames": video,
            "generated_video_path": str(video_path),
            "output_dir": str(output_root),
            **metadata,
        }
        if return_dict:
            return result
        return result["video"]


def _prepend_sys_path(path_value: str | Path) -> None:
    resolved = str(Path(path_value).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _ensure_canonical_diffsynth() -> None:
    _prepend_sys_path(DIFFSYNTH_PARENT)


def _load_runtime_objects(runtime_root: str):
    _prepend_sys_path(runtime_root)
    _ensure_canonical_diffsynth()

    # Legacy external runtime may import diffsynth.core.data; no worldfoundry.core.data equivalent.
    sys.modules.setdefault("diffsynth.core.data", types.ModuleType("diffsynth.core.data"))
    if "diffsynth.utils.data" not in sys.modules:
        data_module = types.ModuleType("diffsynth.utils.data")

        def save_video(frames, path, fps=24, quality=8, ffmpeg_params=None):
            del quality, ffmpeg_params
            save_video_frames(frames, path, fps=int(fps))

        data_module.save_video = save_video
        sys.modules["diffsynth.utils.data"] = data_module
    from diffsynth.pipelines.wan_video_ittakestwo import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video
    from utils import load_config

    return ModelConfig, WanVideoPipeline, load_config, save_video


def _to_torch_tree(value: Any, device: str):
    if isinstance(value, dict):
        return {key: _to_torch_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_torch_tree(child, device) for child in value]
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=torch.bfloat16)
    return tensor.to(device=device, dtype=torch.long)


def _derive_env_obv_from_image(runtime_root: str, input_image_path: str) -> torch.Tensor:
    _prepend_sys_path(runtime_root)
    _ensure_canonical_diffsynth()
    from diffsynth.models.wan_env_preprocess import load_and_preprocess_images

    left_view = load_and_preprocess_images(
        [input_image_path],
        mode="pad",
        return_view="left",
    )[None, None, ...]
    right_view = load_and_preprocess_images(
        [input_image_path],
        mode="pad",
        return_view="right",
    )[None, None, ...]
    return torch.cat([left_view, right_view], dim=2)


def _resolve_output_width(eval_dataset_params, width_override: int | None) -> int:
    if width_override is not None:
        return int(width_override)
    base_width = int(eval_dataset_params.video_params.width)
    return_view = str(getattr(eval_dataset_params, "return_view", "both"))
    if return_view == "both":
        return base_width
    return base_width // 2


def _resolve_num_frames(action_tree: dict[str, Any], fallback: int) -> int:
    for key in ("discrete_action", "continuous_action"):
        value = action_tree.get(key)
        if hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[1])
    return int(fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-sample MultiWorld ItTakesTwo runner.")
    parser.add_argument("--runtime_root", required=True, type=str)
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--input_image_path", required=True, type=str)
    parser.add_argument("--action_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--save_name", default="multiworld_ittakestwo", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--env_obv_path", default=None, type=str)
    parser.add_argument("--num_frames", default=None, type=int)
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--fps", default=None, type=int)
    parser.add_argument("--num_inference_steps", default=35, type=int)
    parser.add_argument("--inference_seed", default=0, type=int)
    parser.add_argument("--derive_env_obv_from_image", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = str(Path(args.runtime_root).expanduser().resolve())
    if args.device.startswith("cuda"):
        if args.device == "cuda":
            args.device = "cuda:0"
        torch.cuda.set_device(args.device)

    ModelConfig, WanVideoPipeline, load_config, save_video = _load_runtime_objects(runtime_root)
    config = load_config(args.config_path)

    wan_root = Path(resolve_wan_ti2v_root())
    wan_model_id = wan_root.name
    wan_local_model_path = str(wan_root.parent)
    wan_dit_paths = [str(wan_root / filename) for filename in WAN_TI2V_DIT_FILENAMES]
    config.simulator_config.dit_config.model_path = wan_dit_paths

    pipe = WanVideoPipeline.from_pretrained(
        config=config,
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(
                local_model_path=wan_local_model_path,
                model_id=wan_model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                skip_download=True,
            ),
            ModelConfig(
                local_model_path=wan_local_model_path,
                model_id=wan_model_id,
                origin_file_pattern="Wan2.2_VAE.pth",
                skip_download=True,
            ),
        ],
    )
    pipe.load_from_checkpoint([*wan_dit_paths, args.checkpoint_path])
    if getattr(pipe, "env_encoder", None) is not None:
        pipe.env_encoder.to(args.device)

    action_tree = load_tree(args.action_path)
    action = _to_torch_tree(action_tree, args.device)

    if args.env_obv_path:
        env_obv = _to_torch_tree(load_tree(args.env_obv_path), args.device)
    elif args.derive_env_obv_from_image:
        env_obv = _derive_env_obv_from_image(runtime_root, args.input_image_path).to(
            args.device,
            dtype=torch.bfloat16,
        )
    else:
        env_obv = None

    eval_dataset_params = config.eval_dataset_config.params
    num_frames = args.num_frames or _resolve_num_frames(
        action_tree,
        int(eval_dataset_params.video_params.num_frames),
    )
    height = int(args.height or eval_dataset_params.video_params.height)
    width = _resolve_output_width(eval_dataset_params, args.width)
    fps = int(args.fps or max(1, int(60 // int(eval_dataset_params.video_params.frame_skip))))

    generated_video = pipe(
        input_image=Image.open(args.input_image_path).convert("RGB"),
        action=action,
        env_obv=env_obv,
        seed=int(args.inference_seed),
        tiled=False,
        height=height,
        width=width,
        num_frames=int(num_frames),
        num_inference_steps=int(args.num_inference_steps),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.save_name}.mp4"
    metadata_path = output_dir / f"{args.save_name}.json"

    save_video(generated_video, str(video_path), fps=fps, quality=10)
    metadata_path.write_text(
        json.dumps(
            {
                "generated_video_path": str(video_path),
                "config_path": str(Path(args.config_path).expanduser().resolve()),
                "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
                "runtime_root": runtime_root,
                "device": args.device,
                "num_frames": int(num_frames),
                "height": height,
                "width": width,
                "fps": fps,
                "num_inference_steps": int(args.num_inference_steps),
                "inference_seed": int(args.inference_seed),
                "derived_env_obv_from_image": bool(
                    args.derive_env_obv_from_image and args.env_obv_path is None
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "MultiWorldItTakesTwoRuntime",
    "dump_tree",
    "load_ittakestwo_action_csv",
    "load_pil_image",
    "load_tree",
    "load_video_frames",
    "main",
    "save_image_input",
]


if __name__ == "__main__":
    raise SystemExit(main())
