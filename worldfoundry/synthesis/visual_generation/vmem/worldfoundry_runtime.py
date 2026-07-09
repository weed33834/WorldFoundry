from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.synthesis.visual_generation.vmem.runtime_env import (
    default_config_path,
    ensure_vmem_runtime,
)


DEFAULT_VMEM_REPO = "liguang0115/vmem"
DEFAULT_VMEM_SURFEL_REPO = "liguang0115/cut3r"
OPENCLIP_VITH14_NAMES = {
    "laion2b_s32b_b79k",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


def _local_weight_candidates(repo_or_path: str, filename: str) -> list[Path]:
    value = str(repo_or_path or "").strip()
    candidates: list[Path] = []
    if value:
        path = Path(value).expanduser()
        if path.is_file():
            candidates.append(path)
        candidates.append(path / filename)

    if value == DEFAULT_VMEM_REPO:
        candidates.append(checkpoint_root_path("vmem", filename))
    elif value == DEFAULT_VMEM_SURFEL_REPO:
        candidates.append(checkpoint_root_path("cut3r", filename))
    return candidates


def _local_weight_file(repo_or_path: str, filename: str) -> Optional[Path]:
    for candidate in _local_weight_candidates(repo_or_path, filename):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _local_sd21_base_path() -> Optional[Path]:
    candidates = [
        checkpoint_root_path("stable-diffusion-2-1-base"),
        checkpoint_root_path("hfd", "stabilityai--stable-diffusion-2-1-base"),
        checkpoint_root_path("models--stabilityai--stable-diffusion-2-1-base"),
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if (resolved / "vae" / "config.json").is_file():
            return resolved
    return None


def _local_open_clip_pretrained(model_name: Any, pretrained: Any) -> Optional[Path]:
    if str(model_name) != "ViT-H-14" or str(pretrained) not in OPENCLIP_VITH14_NAMES:
        return None
    for filename in ("open_clip_pytorch_model.bin", "open_clip_model.safetensors"):
        candidate = checkpoint_root_path("CLIP-ViT-H-14-laion2B-s32B-b79K", filename)
        if candidate.is_file():
            return candidate
    return None


@contextmanager
def _prefer_local_open_clip():
    try:
        import open_clip
    except Exception:
        yield
        return

    original_create_model_and_transforms = open_clip.create_model_and_transforms

    def _worldfoundry_create_model_and_transforms(model_name, *args, **kwargs):
        args_list = list(args)
        pretrained = kwargs.get("pretrained", args_list[0] if args_list else None)
        local_file = _local_open_clip_pretrained(model_name, pretrained)
        if local_file is not None:
            if "pretrained" in kwargs:
                kwargs["pretrained"] = str(local_file)
            elif args_list:
                args_list[0] = str(local_file)
            else:
                kwargs["pretrained"] = str(local_file)
        return original_create_model_and_transforms(model_name, *args_list, **kwargs)

    open_clip.create_model_and_transforms = _worldfoundry_create_model_and_transforms
    try:
        yield
    finally:
        open_clip.create_model_and_transforms = original_create_model_and_transforms


@contextmanager
def _prefer_local_hf_weights(model_ref: str, surfel_ref: str):
    import huggingface_hub

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def _hf_hub_download(repo_id, filename, *args, **kwargs):
        filename_str = str(filename)
        repo_id_str = str(repo_id)
        if filename_str == "vmem_weights.pth":
            local_file = _local_weight_file(repo_id_str, filename_str)
        elif filename_str == "cut3r_512_dpt_4_64.pth":
            local_file = _local_weight_file(repo_id_str, filename_str)
        else:
            local_file = None
        if local_file is not None:
            return str(local_file)
        return original_hf_hub_download(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = _hf_hub_download
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original_hf_hub_download


@contextmanager
def _prefer_local_sd21_vae():
    import torch.nn as nn

    from modeling.modules import autoencoder as autoencoder_module

    local_base = _local_sd21_base_path()
    if local_base is None:
        yield
        return

    original_init = autoencoder_module.AutoEncoder.__init__

    def _worldfoundry_autoencoder_init(self, chunk_size: int):
        nn.Module.__init__(self)
        self.module = autoencoder_module.AutoencoderKL.from_pretrained(
            str(local_base),
            subfolder="vae",
            force_download=False,
            low_cpu_mem_usage=False,
        )
        self.module.eval().requires_grad_(False)
        self.chunk_size = chunk_size

    autoencoder_module.AutoEncoder.__init__ = _worldfoundry_autoencoder_init
    try:
        yield
    finally:
        autoencoder_module.AutoEncoder.__init__ = original_init


def _to_pil_image(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, str):
        return Image.open(data).convert("RGB")
    if isinstance(data, np.ndarray):
        array = np.asarray(data)
        if array.ndim == 4:
            array = array[0]
        if array.dtype in (np.float16, np.float32, np.float64):
            if array.min() >= -1.0 and array.max() <= 1.0:
                if array.min() < 0.0:
                    array = (array + 1.0) * 127.5
                else:
                    array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array).convert("RGB")
    if isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        return _to_pil_image(tensor.numpy())
    raise TypeError(f"Unsupported VMem input type: {type(data)!r}")


class VMemRuntime:
    """Stateful wrapper around the official VMem runtime."""

    def __init__(
        self,
        runtime_pipeline,
        config,
        *,
        transform_img_and_K,
        get_default_intrinsics,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.float32,
        step_size: float = 0.1,
        num_interpolation_frames: int = 4,
    ):
        self.runtime_pipeline = runtime_pipeline
        self.config = config
        self.transform_img_and_K = transform_img_and_K
        self.get_default_intrinsics = get_default_intrinsics
        self.device = device
        self.weight_dtype = weight_dtype
        self.step_size = float(step_size)
        self.num_interpolation_frames = int(num_interpolation_frames)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        args=None,
        device: Optional[str] = None,
        weight_dtype: torch.dtype = torch.float32,
        config_path: Optional[str] = None,
        surfel_model_path: Optional[str] = None,
        step_size: float = 0.1,
        num_interpolation_frames: int = 4,
        runtime_root: Optional[str] = None,
        visualization_dir: Optional[str] = None,
        **kwargs,
    ) -> "VMemRuntime":
        ensure_vmem_runtime(runtime_root)

        from modeling.pipeline import VMemPipeline as RuntimeVMemPipeline
        from omegaconf import OmegaConf
        from utils import get_default_intrinsics, transform_img_and_K

        resolved_config_path = default_config_path(runtime_root) if config_path is None else config_path
        config = OmegaConf.load(str(resolved_config_path)) if args is None else deepcopy(args)

        config.model.model_path = pretrained_model_path or DEFAULT_VMEM_REPO
        config.surfel.model_path = surfel_model_path or DEFAULT_VMEM_SURFEL_REPO

        if visualization_dir:
            config.model.samples_dir = visualization_dir
            config.visualization_dir = visualization_dir

        for key in [
            "height",
            "width",
            "original_height",
            "original_width",
            "context_num_frames",
            "target_num_frames",
            "num_frames",
            "inference_num_steps",
            "cfg",
            "cfg_min",
            "guider_types",
            "camera_scale",
            "translation_distance_weight",
            "use_non_maximum_suppression",
        ]:
            if key in kwargs:
                config.model[key] = kwargs.pop(key)
        for key in [
            "use_surfel",
            "shrink_factor",
            "radius_scale",
            "conf_thresh",
            "merge_position_threshold",
            "merge_normal_threshold",
            "lr",
            "niter",
            "width",
            "height",
        ]:
            if key in kwargs:
                config.surfel[key] = kwargs.pop(key)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with (
            _prefer_local_hf_weights(str(config.model.model_path), str(config.surfel.model_path)),
            _prefer_local_open_clip(),
            _prefer_local_sd21_vae(),
        ):
            runtime_pipeline = RuntimeVMemPipeline(
                config=config,
                device=device,
                dtype=weight_dtype,
            )
        return cls(
            runtime_pipeline=runtime_pipeline,
            config=config,
            transform_img_and_K=transform_img_and_K,
            get_default_intrinsics=get_default_intrinsics,
            device=device,
            weight_dtype=weight_dtype,
            step_size=step_size,
            num_interpolation_frames=num_interpolation_frames,
        )

    @property
    def height(self) -> int:
        return int(self.config.model.height)

    @property
    def width(self) -> int:
        return int(self.config.model.width)

    @property
    def fps(self) -> int:
        return 13

    def is_initialized(self) -> bool:
        return self.current_pose is not None and len(self.frames) > 0

    def reset(self):
        self.runtime_pipeline.reset()
        self.current_pose = None
        self.current_K = None
        self.frames: list[Image.Image] = []
        self.pose_history: list[dict[str, Any]] = []

    def _prepare_image_tensor(self, image: Any) -> torch.Tensor:
        pil_image = _to_pil_image(image)
        image_array = np.asarray(pil_image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
        image_tensor, _ = self.transform_img_and_K(
            image_tensor,
            (self.width, self.height),
            mode="crop",
            K=None,
        )
        image_tensor = image_tensor.to(self.device) * 2.0 - 1.0
        return image_tensor

    def _default_pose(self) -> np.ndarray:
        return np.eye(4, dtype=np.float32)

    def _default_intrinsics(self) -> np.ndarray:
        return np.asarray(
            self.get_default_intrinsics()[0].detach().cpu().numpy(),
            dtype=np.float32,
        )

    def _initialize_state(self, image: Any):
        image_tensor = self._prepare_image_tensor(image)
        initial_pose = self._default_pose()
        initial_K = self._default_intrinsics()
        initial_frame = self.runtime_pipeline.initialize(image_tensor, initial_pose, initial_K)
        self.current_pose = initial_pose
        self.current_K = initial_K
        self.frames = [initial_frame.convert("RGB")]
        self.pose_history = [
            {
                "file_path": "images/frame_001.png",
                "transform_matrix": initial_pose.tolist(),
            }
        ]

    def _interpolate_poses(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        start_rotation = start_pose[:3, :3]
        end_rotation = end_pose[:3, :3]
        start_translation = start_pose[:3, 3]
        end_translation = end_pose[:3, 3]

        import scipy.spatial.transform as spt

        slerp = spt.Slerp(
            np.array([0.0, 1.0]),
            spt.Rotation.from_quat(
                [
                    spt.Rotation.from_matrix(start_rotation).as_quat(),
                    spt.Rotation.from_matrix(end_rotation).as_quat(),
                ]
            ),
        )

        interpolated = []
        for frame_idx in range(num_frames):
            alpha = (frame_idx + 1) / num_frames
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = slerp(alpha).as_matrix().astype(np.float32)
            pose[:3, 3] = (
                (1.0 - alpha) * start_translation + alpha * end_translation
            ).astype(np.float32)
            interpolated.append(pose)
        return interpolated

    def _execute_trajectory(self, poses: Sequence[np.ndarray]) -> list[Image.Image]:
        if not poses:
            return []
        intrinsics = [self.current_K] * len(poses)
        new_frames = self.runtime_pipeline.generate_trajectory_frames(
            list(poses),
            intrinsics,
            use_non_maximum_suppression=False,
        )
        new_frames = [frame.convert("RGB") for frame in new_frames]
        self.current_pose = poses[-1]
        self.frames.extend(new_frames)
        self.pose_history.append(
            {
                "file_path": f"images/frame_{len(self.pose_history) + 1:03d}.png",
                "transform_matrix": self.current_pose.tolist(),
            }
        )
        return new_frames

    def _move_along_view(self, direction: float, num_steps: int = 1) -> list[Image.Image]:
        forward_dir = self.current_pose[:3, 2]
        target_pose = self.current_pose.copy()
        target_pose[:3, 3] += forward_dir * self.step_size * num_steps * direction
        poses = self._interpolate_poses(
            self.current_pose,
            target_pose,
            self.num_interpolation_frames,
        )
        return self._execute_trajectory(poses)

    def _turn(self, degrees: float) -> list[Image.Image]:
        angle_rad = np.radians(degrees)
        rotation = np.array(
            [
                [np.cos(angle_rad), 0.0, np.sin(angle_rad), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-np.sin(angle_rad), 0.0, np.cos(angle_rad), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[:3, :3] = rotation[:3, :3] @ self.current_pose[:3, :3]
        target_pose[:3, 3] = self.current_pose[:3, 3]
        poses = self._interpolate_poses(
            self.current_pose,
            target_pose,
            self.num_interpolation_frames,
        )
        return self._execute_trajectory(poses)

    def _apply_command(self, command: str):
        command = {
            "forward": "w",
            "backward": "s",
            "camera_l": "a",
            "camera_left": "a",
            "left": "a",
            "camera_r": "d",
            "camera_right": "d",
            "right": "d",
        }.get(command, command)
        if command == "w":
            self._move_along_view(direction=-1.0, num_steps=1)
        elif command == "s":
            self._move_along_view(direction=1.0, num_steps=1)
        elif command == "a":
            self._turn(4.0)
        elif command == "d":
            self._turn(-4.0)
        else:
            raise ValueError(f"Unsupported VMem command: {command}")

    @torch.no_grad()
    def predict(
        self,
        *,
        image: Optional[Any] = None,
        actions: Optional[Sequence[str]] = None,
        reset_state: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, Any] | np.ndarray:
        del kwargs

        if reset_state:
            self.reset()

        if image is not None or not self.is_initialized():
            if image is None:
                raise ValueError("An input image is required to initialize VMem.")
            self._initialize_state(image)

        commands = [str(action) for action in (actions or []) if str(action).strip()]
        for command in commands:
            self._apply_command(command)

        video = np.stack([np.asarray(frame, dtype=np.uint8) for frame in self.frames], axis=0)
        result = {
            "video": video,
            "frames": list(self.frames),
            "last_frame": self.frames[-1].copy(),
            "pose_history": deepcopy(self.pose_history),
            "num_frames": int(video.shape[0]),
            "fps": self.fps,
        }
        if return_dict:
            return result
        return result["video"]


__all__ = ["DEFAULT_VMEM_REPO", "DEFAULT_VMEM_SURFEL_REPO", "VMemRuntime"]
