"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime_env import (
    DEFAULT_GEN3C_MOGE1_REPO,
    DEFAULT_GEN3C_NEGATIVE_PROMPT,
    build_subprocess_env,
    load_pil_image,
    load_video_frames,
    prepare_gen3c_checkpoint_root,
    project_root,
    resolve_gen3c_checkpoint_arg,
    resolve_gen3c_runtime_root,
    resolve_gen3c_moge_pretrained,
)


class Gen3CRuntime:
    """Subprocess wrapper around the packaged GEN3C single-image inference path."""

    def __init__(
        self,
        model_root: str,
        checkpoint_dir: str,
        moge_pretrained: str,
        device: str = "cuda",
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """Init.

        Args:
            model_root: The model root.
            checkpoint_dir: The checkpoint dir.
            moge_pretrained: The moge pretrained.
            device: The device.
            defaults: The defaults.
        """
        self.model_root = model_root
        self.checkpoint_dir = checkpoint_dir
        self.moge_pretrained = moge_pretrained
        self.device = device
        self.defaults = defaults or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str],
        args=None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        moge_path: Optional[str] = None,
        moge_pretrained: Optional[str] = None,
        default_trajectory: str = "left",
        default_camera_rotation: str = "center_facing",
        default_movement_distance: float = 0.3,
        guidance: float = 1.0,
        num_steps: int = 35,
        num_video_frames: int = 121,
        fps: int = 24,
        height: int = 704,
        width: int = 1280,
        seed: int = 1,
        noise_aug_strength: float = 0.0,
        save_buffer: bool = False,
        filter_points_threshold: float = 0.05,
        foreground_masking: bool = True,
        disable_prompt_upsampler: bool = True,
        disable_guardrail: bool = True,
        disable_prompt_encoder: bool = False,
        offload_diffusion_transformer: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        num_gpus: int = 1,
        negative_prompt: str = DEFAULT_GEN3C_NEGATIVE_PROMPT,
        **kwargs,
    ) -> "Gen3CRuntime":
        """From pretrained.

        Args:
            pretrained_model_path: The pretrained model path.
            args: The args.
            device: The device.
            checkpoint_dir: The checkpoint dir.
            moge_path: The moge path.
            moge_pretrained: The moge pretrained.
            default_trajectory: The default trajectory.
            default_camera_rotation: The default camera rotation.
            default_movement_distance: The default movement distance.
            guidance: The guidance.
            num_steps: The num steps.
            num_video_frames: The num video frames.
            fps: The fps.
            height: The height.
            width: The width.
            seed: The seed.
            noise_aug_strength: The noise aug strength.
            save_buffer: The save buffer.
            filter_points_threshold: The filter points threshold.
            foreground_masking: The foreground masking.
            disable_prompt_upsampler: The disable prompt upsampler.
            disable_guardrail: The disable guardrail.
            disable_prompt_encoder: The disable prompt encoder.
            offload_diffusion_transformer: The offload diffusion transformer.
            offload_tokenizer: The offload tokenizer.
            offload_text_encoder_model: The offload text encoder model.
            offload_prompt_upsampler: The offload prompt upsampler.
            offload_guardrail_models: The offload guardrail models.
            num_gpus: The num gpus.
            negative_prompt: The negative prompt.

        Returns:
            The return value.
        """
        del args
        model_root = resolve_gen3c_runtime_root()
        checkpoint_arg = checkpoint_dir or resolve_gen3c_checkpoint_arg(pretrained_model_path)
        if moge_path is not None and str(moge_path).strip():
            raise ValueError("MoGe runtime is packaged in-tree; pass `moge_pretrained` for weights only.")
        resolved_checkpoint_dir = prepare_gen3c_checkpoint_root(checkpoint_arg)
        resolved_moge_pretrained = resolve_gen3c_moge_pretrained(
            moge_pretrained or DEFAULT_GEN3C_MOGE1_REPO
        )
        defaults = {
            "default_trajectory": default_trajectory,
            "default_camera_rotation": default_camera_rotation,
            "default_movement_distance": default_movement_distance,
            "guidance": guidance,
            "num_steps": num_steps,
            "num_video_frames": num_video_frames,
            "fps": fps,
            "height": height,
            "width": width,
            "seed": seed,
            "noise_aug_strength": noise_aug_strength,
            "save_buffer": save_buffer,
            "filter_points_threshold": filter_points_threshold,
            "foreground_masking": foreground_masking,
            "disable_prompt_upsampler": disable_prompt_upsampler,
            "disable_guardrail": disable_guardrail,
            "disable_prompt_encoder": disable_prompt_encoder,
            "offload_diffusion_transformer": offload_diffusion_transformer,
            "offload_tokenizer": offload_tokenizer,
            "offload_text_encoder_model": offload_text_encoder_model,
            "offload_prompt_upsampler": offload_prompt_upsampler,
            "offload_guardrail_models": offload_guardrail_models,
            "num_gpus": num_gpus,
            "negative_prompt": negative_prompt,
            **kwargs,
        }
        return cls(
            model_root=model_root,
            checkpoint_dir=resolved_checkpoint_dir,
            moge_pretrained=resolved_moge_pretrained,
            device=device or "cuda",
            defaults=defaults,
        )

    def predict(
        self,
        image,
        prompt: str = "",
        trajectory: Optional[str] = None,
        camera_rotation: Optional[str] = None,
        movement_distance: Optional[float] = None,
        output_dir: Optional[str] = None,
        scene_name: str = "gen3c_scene",
        negative_prompt: Optional[str] = None,
        return_dict: bool = False,
        show_progress: bool = True,
        **kwargs,
    ):
        """Predict.

        Args:
            image: The image.
            prompt: The prompt.
            trajectory: The trajectory.
            camera_rotation: The camera rotation.
            movement_distance: The movement distance.
            output_dir: The output dir.
            scene_name: The scene name.
            negative_prompt: The negative prompt.
            return_dict: The return dict.
            show_progress: The show progress.
        """
        output_root = (
            Path(output_dir).expanduser().resolve() / scene_name
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="gen3c_")).resolve() / scene_name
        )
        output_root.mkdir(parents=True, exist_ok=True)

        source_image_path = (
            Path(image).expanduser()
            if isinstance(image, (str, os.PathLike)) and Path(image).expanduser().is_file()
            else None
        )
        if source_image_path is not None:
            suffix = source_image_path.suffix or ".png"
            image_path = output_root / f"input{suffix}"
            shutil.copy2(source_image_path, image_path)
        else:
            image_rgb = load_pil_image(image)
            image_path = output_root / "input.png"
            image_rgb.save(image_path)

        trajectory = str(
            trajectory
            or kwargs.pop("default_trajectory", None)
            or self.defaults.get("default_trajectory", "left")
        )
        camera_rotation = str(
            camera_rotation
            or self.defaults.get("default_camera_rotation", "center_facing")
        )
        movement_distance = float(
            movement_distance
            if movement_distance is not None
            else self.defaults.get("default_movement_distance", 0.3)
        )

        num_gpus = int(kwargs.pop("num_gpus", self.defaults.get("num_gpus", 1)))
        # Remove upstream kwargs that collide with _build_command's explicit params
        kwargs.pop("image_path", None)
        kwargs.pop("input_path", None)
        command = self._build_command(
            output_root=output_root,
            image_path=image_path,
            prompt=prompt or "",
            negative_prompt=negative_prompt
            if negative_prompt is not None
            else self.defaults.get("negative_prompt", DEFAULT_GEN3C_NEGATIVE_PROMPT),
            trajectory=trajectory,
            camera_rotation=camera_rotation,
            movement_distance=movement_distance,
            num_gpus=num_gpus,
            **kwargs,
        )

        env = build_subprocess_env(device=None if num_gpus > 1 else self.device)
        log_path = output_root / "gen3c_runner.log"
        completed = subprocess.run(
            command,
            cwd=project_root(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if show_progress and completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-80:])
            raise RuntimeError(
                f"GEN3C runner exited with status {completed.returncode}; "
                f"see {log_path}\n{tail}"
            )

        result_path = output_root / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"GEN3C result metadata not found: {result_path}")
        with result_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        generated_video_path = metadata["generated_video_path"]
        video = load_video_frames(generated_video_path)
        result = {
            "frames": video,
            "video": video,
            "generated_video_path": generated_video_path,
            "output_dir": metadata.get("output_dir"),
            "scene_name": metadata.get("scene_name", scene_name),
            "trajectory": metadata.get("trajectory", trajectory),
            "camera_rotation": metadata.get("camera_rotation", camera_rotation),
            "movement_distance": metadata.get("movement_distance", movement_distance),
            "fps": int(metadata.get("fps", self.defaults.get("fps", 24))),
            "prompt": metadata.get("prompt", prompt or ""),
            "negative_prompt": metadata.get("negative_prompt"),
            "num_video_frames": int(
                metadata.get("num_video_frames", self.defaults.get("num_video_frames", 121))
            ),
            "checkpoint_dir": self.checkpoint_dir,
            "model_root": self.model_root,
            "moge_pretrained": self.moge_pretrained,
        }
        if return_dict:
            return result
        return result["video"]

    def _build_command(
        self,
        *,
        output_root: Path,
        image_path: Path,
        prompt: str,
        negative_prompt: str,
        trajectory: str,
        camera_rotation: str,
        movement_distance: float,
        num_gpus: int,
        **kwargs,
    ) -> list[str]:
        """Helper function to build command.

        Returns:
            The return value.
        """
        command = [
            sys.executable,
            "-m",
            "worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c.worldfoundry_runner",
        ]
        if num_gpus > 1:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(num_gpus),
                "-m",
                "worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c.worldfoundry_runner",
            ]

        def _bool_arg(name: str, value: bool) -> list[str]:
            """Helper function to bool arg.

            Args:
                name: The name.
                value: The value.

            Returns:
                The return value.
            """
            return [name] if value else []

        command.extend(
            [
                "--checkpoint_dir",
                self.checkpoint_dir,
                "--input_image_path",
                str(image_path),
                "--output_dir",
                str(output_root),
                "--scene_name",
                output_root.name,
                "--video_save_name",
                "video",
                "--prompt",
                prompt,
                "--negative_prompt",
                negative_prompt,
                "--trajectory",
                trajectory,
                "--camera_rotation",
                camera_rotation,
                "--movement_distance",
                str(float(movement_distance)),
                "--guidance",
                str(float(kwargs.get("guidance", self.defaults.get("guidance", 1.0)))),
                "--num_steps",
                str(int(kwargs.get("num_steps", self.defaults.get("num_steps", 35)))),
                "--num_video_frames",
                str(
                    int(
                        kwargs.get(
                            "num_video_frames",
                            self.defaults.get("num_video_frames", 121),
                        )
                    )
                ),
                "--fps",
                str(int(kwargs.get("fps", self.defaults.get("fps", 24)))),
                "--height",
                str(int(kwargs.get("height", self.defaults.get("height", 704)))),
                "--width",
                str(int(kwargs.get("width", self.defaults.get("width", 1280)))),
                "--seed",
                str(int(kwargs.get("seed", self.defaults.get("seed", 1)))),
                "--noise_aug_strength",
                str(
                    float(
                        kwargs.get(
                            "noise_aug_strength",
                            self.defaults.get("noise_aug_strength", 0.0),
                        )
                    )
                ),
                "--filter_points_threshold",
                str(
                    float(
                        kwargs.get(
                            "filter_points_threshold",
                            self.defaults.get("filter_points_threshold", 0.05),
                        )
                    )
                ),
                "--num_gpus",
                str(int(num_gpus)),
                "--moge_pretrained",
                str(kwargs.get("moge_pretrained", self.moge_pretrained)),
            ]
        )

        command.extend(
            _bool_arg(
                "--save_buffer",
                bool(kwargs.get("save_buffer", self.defaults.get("save_buffer", False))),
            )
        )
        command.extend(
            _bool_arg(
                "--foreground_masking",
                bool(
                    kwargs.get(
                        "foreground_masking",
                        self.defaults.get("foreground_masking", True),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--disable_prompt_upsampler",
                bool(
                    kwargs.get(
                        "disable_prompt_upsampler",
                        self.defaults.get("disable_prompt_upsampler", True),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--disable_guardrail",
                bool(
                    kwargs.get(
                        "disable_guardrail",
                        self.defaults.get("disable_guardrail", True),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--disable_prompt_encoder",
                bool(
                    kwargs.get(
                        "disable_prompt_encoder",
                        self.defaults.get("disable_prompt_encoder", False),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--offload_diffusion_transformer",
                bool(
                    kwargs.get(
                        "offload_diffusion_transformer",
                        self.defaults.get("offload_diffusion_transformer", False),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--offload_tokenizer",
                bool(
                    kwargs.get(
                        "offload_tokenizer",
                        self.defaults.get("offload_tokenizer", False),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--offload_text_encoder_model",
                bool(
                    kwargs.get(
                        "offload_text_encoder_model",
                        self.defaults.get("offload_text_encoder_model", False),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--offload_prompt_upsampler",
                bool(
                    kwargs.get(
                        "offload_prompt_upsampler",
                        self.defaults.get("offload_prompt_upsampler", False),
                    )
                ),
            )
        )
        command.extend(
            _bool_arg(
                "--offload_guardrail_models",
                bool(
                    kwargs.get(
                        "offload_guardrail_models",
                        self.defaults.get("offload_guardrail_models", False),
                    )
                ),
            )
        )
        return command


__all__ = ["Gen3CRuntime"]
