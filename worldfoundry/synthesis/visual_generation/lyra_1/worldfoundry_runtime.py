from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from worldfoundry.core.io.paths import package_module_root as package_root


class Lyra1Runtime:
    """
    WorldFoundry adapter around the packaged Lyra-1 Scene-Generation Diffusion (SDG) runtime sources.

    This class provides an interface to configure and execute the Lyra-1 model,
    handling checkpoint management, command building, input preparation,
    and output location.
    """

    MODEL_ID = "lyra-1"
    DISPLAY_NAME = "Lyra-1"
    BLOCKED_REASONS = (
        "Full Lyra-1 execution requires the official CUDA stack, Cosmos/MoGe dependencies, and local checkpoints.",
        "The source runtime lives under synthesis/visual_generation; checkpoints remain external caller-provided assets.",
    )

    # Maps human-readable trajectory names to their corresponding numerical indices
    # used by the underlying Lyra-1 runtime for multi-trajectory generation.
    MULTI_TRAJECTORY_INDEX = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 2,  # 'down' often maps to the same trajectory index as 'up'
        "zoom_out": 3,
        "zoom_in": 4,
        "clockwise": 5,
        "counterclockwise": 5,  # 'counterclockwise' often maps to the same trajectory index as 'clockwise'
    }

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cuda",
        defaults: Optional[dict[str, Any]] = None,
        model_id: str = MODEL_ID,
    ) -> None:
        """
        Initialize a Lyra-1 in-tree wrapper.

        Args:
            checkpoint_dir: Optional external Lyra-1/Cosmos checkpoint directory.
            device: Target device label (e.g., "cuda", "cpu") preserved in generated plans.
            defaults: Default sampling and execution options, overriding internal defaults.
            model_id: Runtime profile identifier, defaults to `Lyra1Runtime.MODEL_ID`.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "image_or_video_to_world_video"
        # Resolve and expand the checkpoint directory path
        self.checkpoint_dir = None if checkpoint_dir is None else str(Path(checkpoint_dir).expanduser())
        self.device = device
        # Initialize defaults, allowing caller overrides
        self.defaults = defaults or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        checkpoint_dir: Optional[str] = None,
        default_mode: str = "static",
        **kwargs: Any,
    ) -> "Lyra1Runtime":
        """
        Build a lazy Lyra-1 in-tree runtime wrapper.

        This factory method allows initializing the runtime from a path or a mapping of options.

        Args:
            pretrained_model_path: Optional checkpoint directory path or a mapping with runtime options.
            args: Unused compatibility argument for factory callers (e.g., HuggingFace `from_pretrained` style).
            device: Target device label (e.g., "cuda", "cpu").
            checkpoint_dir: Explicit external checkpoint directory, takes precedence over `pretrained_model_path` if both are provided.
            default_mode: Default Lyra-1 generation mode, either `static` (image-to-video) or `dynamic` (video-to-video).
            **kwargs: Extra default generation options to be passed to the runtime.
        
        Returns:
            An initialized `Lyra1Runtime` instance.

        Raises:
            FileNotFoundError: If `checkpoint_dir` is not found and cannot be prepared.
        """
        del args  # This argument is ignored for Lyra1Runtime but kept for API compatibility.
        # Parse pretrained_model_path, which can be a path string or a dictionary of options.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)  # Merge additional keyword arguments into options.

        repo_root = options.get("repo_root")
        resolved_checkpoint_dir = checkpoint_dir or options.get("checkpoint_dir")

        # Attempt to prepare the Lyra-1 checkpoint root using worldfoundry utilities.
        try:
            from worldfoundry.pipelines.lyra.lyra_utils import prepare_lyra1_checkpoint_root

            resolved_checkpoint_dir = prepare_lyra1_checkpoint_root(
                checkpoint_dir=resolved_checkpoint_dir,
                repo_root=repo_root,
            )
        except FileNotFoundError:
            # If preparation fails and no checkpoint directory was explicitly provided, re-raise the error.
            if resolved_checkpoint_dir is None:
                raise

        # Instantiate the Lyra1Runtime with resolved parameters.
        return cls(
            checkpoint_dir=resolved_checkpoint_dir,
            device=str(device or options.get("device") or "cuda"),
            defaults={"default_mode": default_mode, **options},
            model_id=str(options.get("model_id") or options.get("profile_id") or cls.MODEL_ID),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """
        Return the absolute path to the Lyra-1 wrapper runtime root directory.

        This directory contains the Lyra-1 specific scripts and entry points.

        Returns:
            A `Path` object pointing to the runtime root.
        """
        return Path(__file__).resolve().parent / "lyra1_runtime"

    @staticmethod
    def _cosmos_root() -> Path:
        """
        Return the absolute path to the Cosmos root directory.

        Cosmos is a dependency for Lyra-1, and its path is resolved via `worldfoundry` package_root.

        Returns:
            A `Path` object pointing to the Cosmos root.
        """
        return package_root(
            "worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c"
        ) / "cosmos_predict1"

    def _script_path(self, mode: str) -> Path:
        """
        Resolve the path to the vendored Lyra-1 generation script based on the specified mode.

        Args:
            mode: Lyra-1 generation mode, "static" for image-to-video or "dynamic" for video-to-video.

        Returns:
            A `Path` object pointing to the specific generation script.
        """
        script_name = "gen3c_single_image_sdg.py" if mode == "static" else "gen3c_dynamic_sdg.py"
        return self._runtime_root() / script_name

    @staticmethod
    def _entry_path() -> Path:
        """
        Return the absolute path to the main worldfoundry entry point script for Lyra-1.

        This script acts as a wrapper to call the actual Lyra-1 generation scripts.

        Returns:
            A `Path` object pointing to the entry point script.
        """
        return Path(__file__).resolve().parent / "lyra1_runtime" / "worldfoundry_entry.py"

    @staticmethod
    def _choose_free_port() -> int:
        """
        Dynamically find and return an available free port on the local machine.

        This is typically used for distributed training (e.g., `torchrun`) to ensure
        communication ports are not in use.

        Returns:
            An integer representing an available port number.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Bind to an available port on localhost (0 means the OS assigns a free port).
            sock.bind(("127.0.0.1", 0))
            # Return the port number assigned by the OS.
            return int(sock.getsockname()[1])

    def _build_command(
        self,
        *,
        mode: str,
        checkpoint_dir: str | Path | None,
        output_root: Path,
        prepared_input_path: str,
        prompt: str,
        trajectory: str,
        num_video_frames: Optional[int],
        fps: Optional[int],
        height: Optional[int],
        width: Optional[int],
        num_steps: Optional[int],
        seed: Optional[int],
        guidance: Optional[float],
        num_gpus: Optional[int],
        movement_distance: Optional[float],
        camera_rotation: Optional[str],
        multi_trajectory: bool,
        total_movement_distance_factor: Optional[float],
        vipe_path: Optional[str],
        vipe_starting_frame_idx: Optional[int],
        filter_points_threshold: Optional[float],
        foreground_masking: Optional[bool],
        center_depth_quantile: Optional[bool],
        flip_supervision: Optional[bool],
        offload_diffusion_transformer: Optional[bool],
        offload_tokenizer: Optional[bool],
        offload_text_encoder_model: Optional[bool],
        offload_prompt_upsampler: Optional[bool],
        offload_guardrail_models: Optional[bool],
        disable_prompt_encoder: Optional[bool],
        disable_guardrail: Optional[bool],
    ) -> list[str]:
        """
        Build the command-line arguments for the Lyra-1 in-tree subprocess.

        This method constructs the full command list, including Python executable,
        entry script, specific Lyra-1 generation script, and all model-specific
        parameters, handling defaults and overrides.

        Args:
            mode: Static (image-to-video) or dynamic (video-to-video) generation mode.
            checkpoint_dir: External checkpoint directory asset. If None, uses `self.checkpoint_dir` or a default "checkpoints".
            output_root: Directory that receives generated files.
            prepared_input_path: Materialized image, video, or VIPE input path.
            prompt: Optional text prompt to guide generation.
            trajectory: Camera trajectory name (e.g., "zoom_in", "left").
            num_video_frames: Number of generated frames for the output video.
            fps: Output frames per second.
            height: Output frame height.
            width: Output frame width.
            num_steps: Number of diffusion inference steps.
            seed: Random seed for reproducible generation.
            guidance: Classifier-free guidance scale.
            num_gpus: Number of GPUs to use. If > 1, `torchrun` is used.
            movement_distance: Camera movement distance.
            camera_rotation: Camera rotation policy (e.g., "center_facing").
            multi_trajectory: Whether to generate all predefined training trajectories in parallel.
            total_movement_distance_factor: Scaling factor for total movement distance when `multi_trajectory` is enabled.
            vipe_path: Optional VIPE (Video Input and Pose Estimation) input path, used for dynamic mode.
            vipe_starting_frame_idx: Index of the reference frame in the VIPE input.
            filter_points_threshold: Warped point filtering threshold.
            foreground_masking: Whether to apply foreground masking.
            center_depth_quantile: Whether to use center depth quantile.
            flip_supervision: Whether to generate flipped supervision.
            offload_diffusion_transformer: Whether to offload the diffusion transformer model.
            offload_tokenizer: Whether to offload the tokenizer.
            offload_text_encoder_model: Whether to offload the text encoder model.
            offload_prompt_upsampler: Whether to offload the prompt upsampler.
            offload_guardrail_models: Whether to offload guardrail models.
            disable_prompt_encoder: Whether to disable the prompt encoder.
            disable_guardrail: Whether to disable guardrail models.

        Returns:
            A list of strings representing the full command to be executed by a subprocess.
        """
        # Resolve checkpoint directory, prioritizing explicit argument, then instance attribute, then a default.
        checkpoint_value = checkpoint_dir or self.checkpoint_dir or "checkpoints"
        # Determine the number of GPUs, prioritizing explicit argument, then instance defaults, then 1.
        resolved_num_gpus = int(num_gpus if num_gpus is not None else self.defaults.get("num_gpus", 1))

        # Initial list of arguments common to both single and multi-GPU execution.
        script_args = [
            "--checkpoint_dir",
            str(Path(checkpoint_value).expanduser()),
            "--video_save_folder",
            str(output_root),
            "--num_gpus",
            str(resolved_num_gpus),
            "--num_video_frames",
            str(int(num_video_frames if num_video_frames is not None else self.defaults.get("num_video_frames", 121))),
            "--num_steps",
            str(int(num_steps if num_steps is not None else self.defaults.get("num_steps", 35))),
            "--height",
            str(int(height if height is not None else self.defaults.get("height", 704))),
            "--width",
            str(int(width if width is not None else self.defaults.get("width", 1280))),
            "--fps",
            str(int(fps if fps is not None else self.defaults.get("fps", 24))),
            "--seed",
            str(int(seed if seed is not None else self.defaults.get("seed", 1))),
            "--guidance",
            str(float(guidance if guidance is not None else self.defaults.get("guidance", 1.0))),
            "--trajectory",
            trajectory,
            "--camera_rotation",
            str(camera_rotation or self.defaults.get("camera_rotation", "center_facing")),
            "--movement_distance",
            str(float(movement_distance if movement_distance is not None else self.defaults.get("movement_distance", 0.3))),
            "--filter_points_threshold",
            str(
                float(
                    filter_points_threshold
                    if filter_points_threshold is not None
                    else self.defaults.get("filter_points_threshold", 0.05)
                )
            ),
        ]

        # Determine the command prefix based on the number of GPUs.
        if resolved_num_gpus > 1:
            # Use 'torchrun' for multi-GPU execution.
            torchrun = Path(sys.executable).with_name("torchrun")
            command = [
                str(torchrun if torchrun.exists() else "torchrun"),  # Use 'torchrun' if found, else rely on PATH
                f"--nproc_per_node={resolved_num_gpus}",
                f"--master_port={self._choose_free_port()}",  # Assign a free port for distributed communication
                str(self._entry_path()),  # WorldFoundry entry point
                "--worldfoundry-script-path",
                str(self._script_path(mode)),  # Actual Lyra-1 generation script
                *script_args,
            ]
        else:
            # Use direct Python execution for single-GPU or CPU.
            command = [
                sys.executable,  # Python executable
                str(self._entry_path()),  # WorldFoundry entry point
                "--worldfoundry-script-path",
                str(self._script_path(mode)),  # Actual Lyra-1 generation script
                *script_args,
            ]

        # Add optional prompt argument if provided.
        if prompt:
            command.extend(["--prompt", prompt])

        # Add input path argument based on mode and VIPE usage.
        if mode == "static":
            command.extend(["--input_image_path", prepared_input_path])
        elif vipe_path is not None:
            command.extend(["--vipe_path", prepared_input_path])
            command.extend(
                [
                    "--vipe_starting_frame_idx",
                    str(int(vipe_starting_frame_idx if vipe_starting_frame_idx is not None else self.defaults.get("vipe_starting_frame_idx", 0))),
                ]
            )
        else:  # Dynamic mode without VIPE, implies video input.
            command.extend(["--input_image_path", prepared_input_path])

        # Add boolean flags if enabled.
        # Iterates through a list of flag names, their explicit values, and their default enabled state.
        for flag_name, flag_value, default_value in [
            ("--foreground_masking", foreground_masking, True),
            ("--multi_trajectory", multi_trajectory, False),
            ("--center_depth_quantile", center_depth_quantile, False),
            ("--flip_supervision", flip_supervision, False),
            ("--offload_diffusion_transformer", offload_diffusion_transformer, False),
            ("--offload_tokenizer", offload_tokenizer, False),
            ("--offload_text_encoder_model", offload_text_encoder_model, False),
            ("--offload_prompt_upsampler", offload_prompt_upsampler, False),
            ("--offload_guardrail_models", offload_guardrail_models, False),
            ("--disable_prompt_encoder", disable_prompt_encoder, False),
            ("--disable_guardrail", disable_guardrail, False),
        ]:
            # Determine if the flag should be enabled: explicit value takes precedence, otherwise check defaults.
            enabled = self.defaults.get(flag_name.lstrip("-"), default_value) if flag_value is None else flag_value
            if bool(enabled):
                command.append(flag_name)

        # Add total_movement_distance_factor argument.
        command.extend(
            [
                "--total_movement_distance_factor",
                str(float(total_movement_distance_factor if total_movement_distance_factor is not None else self.defaults.get("total_movement_distance_factor", 1.0))),
            ]
        )
        return command

    def predict(
        self,
        visual_input: Any = None,
        mode: str = "static",
        prompt: str = "",
        trajectory: str = "zoom_in",
        output_root: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        num_video_frames: Optional[int] = None,
        fps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance: Optional[float] = None,
        num_gpus: Optional[int] = None,
        movement_distance: Optional[float] = None,
        camera_rotation: Optional[str] = None,
        multi_trajectory: bool = False,
        total_movement_distance_factor: Optional[float] = None,
        vipe_path: Optional[str] = None,
        vipe_starting_frame_idx: Optional[int] = None,
        filter_points_threshold: Optional[float] = None,
        foreground_masking: Optional[bool] = None,
        center_depth_quantile: Optional[bool] = None,
        flip_supervision: Optional[bool] = None,
        offload_diffusion_transformer: Optional[bool] = None,
        offload_tokenizer: Optional[bool] = None,
        offload_text_encoder_model: Optional[bool] = None,
        offload_prompt_upsampler: Optional[bool] = None,
        offload_guardrail_models: Optional[bool] = None,
        disable_prompt_encoder: Optional[bool] = None,
        disable_guardrail: Optional[bool] = None,
        show_progress: bool = True,
        execute: bool = False,
        plan_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare or execute a Lyra-1 in-tree runtime plan.

        This method orchestrates the generation process, including input preparation,
        command building, and optional subprocess execution.

        Args:
            visual_input: Static image (Path, PIL.Image.Image, numpy.ndarray) or dynamic video input (Path, list of frames).
            mode: Lyra-1 generation mode, "static" or "dynamic". Defaults to "static".
            prompt: Optional text prompt to guide generation. Defaults to "".
            trajectory: Camera trajectory name (e.g., "zoom_in", "left"). Defaults to "zoom_in".
            output_root: Directory to save generated files. If None, a temporary directory is created.
            checkpoint_dir: External checkpoint directory asset. If None, uses the instance's configured checkpoint_dir.
            num_video_frames: Number of generated frames.
            fps: Output frames per second.
            height: Output frame height.
            width: Output frame width.
            num_steps: Number of diffusion inference steps.
            seed: Random seed for reproducible generation.
            guidance: Classifier-free guidance scale.
            num_gpus: Number of GPUs for the official torchrun path.
            movement_distance: Camera movement distance.
            camera_rotation: Camera rotation policy.
            multi_trajectory: Whether to generate all training trajectories (e.g., for dataset creation).
            total_movement_distance_factor: Multi-trajectory movement scaling factor.
            vipe_path: Optional VIPE input path for dynamic mode.
            vipe_starting_frame_idx: VIPE reference frame index.
            filter_points_threshold: Warped point filtering threshold.
            foreground_masking: Whether to use foreground masking.
            center_depth_quantile: Whether to use center depth quantile.
            flip_supervision: Whether to generate flipped supervision.
            offload_diffusion_transformer: Whether to offload diffusion transformer to CPU.
            offload_tokenizer: Whether to offload tokenizer to CPU.
            offload_text_encoder_model: Whether to offload text encoder to CPU.
            offload_prompt_upsampler: Whether to offload prompt upsampler to CPU.
            offload_guardrail_models: Whether to offload guardrail models to CPU.
            disable_prompt_encoder: Whether to disable prompt encoder.
            disable_guardrail: Whether to disable guardrail models.
            show_progress: If True, subprocess output (stdout/stderr) is streamed to the console. If False, it's redirected to DEVNULL.
            execute: If True, the Lyra-1 script is executed. If False, a plan JSON is generated.
            plan_path: Optional path to save the execution plan JSON. If None, a plan is saved in `output_root`.
            **kwargs: Extra metadata to be preserved in the generated plan.

        Returns:
            A dictionary containing the status of the operation (completed or blocked) and relevant metadata,
            such as generated video path, input path, or the path to the execution plan.

        Raises:
            ValueError: If an unsupported Lyra-1 mode is provided.
            FileNotFoundError: If no generated video is found after execution.
        """
        # Resolve the generation mode, prioritizing explicit argument, then instance defaults, then "static".
        mode = str(mode or self.defaults.get("default_mode", "static")).lower()
        if mode not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported Lyra-1 mode: {mode}")

        # Create or resolve the output root directory. If none is provided, use a temporary directory.
        output_root_path = Path(output_root or tempfile.mkdtemp(prefix=f"lyra1_{mode}_")).expanduser().resolve()
        output_root_path.mkdir(parents=True, exist_ok=True)

        # Prepare the visual input (e.g., materialize image/video to disk if not already a path).
        prepared_input_path = self._prepare_input(
            visual_input=visual_input,
            mode=mode,
            output_root=output_root_path,
            fps=int(fps if fps is not None else self.defaults.get("fps", 24)),
            vipe_path=vipe_path,
            materialize=execute,  # Only materialize input if execution is planned.
        )

        # Build the command for the Lyra-1 subprocess.
        command = self._build_command(
            mode=mode,
            checkpoint_dir=checkpoint_dir,
            output_root=output_root_path,
            prepared_input_path=prepared_input_path,
            prompt=prompt,
            trajectory=trajectory,
            num_video_frames=num_video_frames,
            fps=fps,
            height=height,
            width=width,
            num_steps=num_steps,
            seed=seed,
            guidance=guidance,
            num_gpus=num_gpus,
            movement_distance=movement_distance,
            camera_rotation=camera_rotation,
            multi_trajectory=multi_trajectory,
            total_movement_distance_factor=total_movement_distance_factor,
            vipe_path=vipe_path,
            vipe_starting_frame_idx=vipe_starting_frame_idx,
            filter_points_threshold=filter_points_threshold,
            foreground_masking=foreground_masking,
            center_depth_quantile=center_depth_quantile,
            flip_supervision=flip_supervision,
            offload_diffusion_transformer=offload_diffusion_transformer,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_prompt_encoder=disable_prompt_encoder,
            disable_guardrail=disable_guardrail,
        )

        if execute:
            # Set stdout/stderr behavior based on show_progress flag.
            stdout = None if show_progress else subprocess.DEVNULL
            stderr = None if show_progress else subprocess.STDOUT
            # Copy current environment and set PYTORCH_CUDA_ALLOC_CONF for better memory management.
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            # Execute the Lyra-1 command as a subprocess.
            subprocess.run(command, check=True, cwd=str(self._runtime_root()), env=env, stdout=stdout, stderr=stderr)

            # Locate the generated video file after successful execution.
            generated_video_path = self._locate_generated_video(output_root_path, trajectory, multi_trajectory)
            # Dynamically import lyra_utils to avoid circular dependencies and ensure it's used only when needed.
            from worldfoundry.pipelines.lyra.lyra_utils import load_video_frames

            # Return a dictionary with execution results.
            return {
                "status": "completed",
                "mode": mode,
                "prompt": prompt,
                "trajectory": trajectory,
                "generated_root": str(output_root_path),
                "generated_video_path": generated_video_path,
                "video": load_video_frames(generated_video_path),
                "fps": int(fps if fps is not None else self.defaults.get("fps", 24)),
                "input_path": prepared_input_path,
            }
        
        # If not executing, write an execution plan instead.
        return self._write_plan(
            plan_path=plan_path,
            mode=mode,
            output_root=output_root_path,
            prepared_input_path=prepared_input_path,
            command=command,
            prompt=prompt,
            trajectory=trajectory,
            checkpoint_dir=checkpoint_dir,
            extra=kwargs,
        )

    def _write_plan(
        self,
        *,
        plan_path: str | Path | None,
        mode: str,
        output_root: Path,
        prepared_input_path: str,
        command: list[str],
        prompt: str,
        trajectory: str,
        checkpoint_dir: Optional[str],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Write a Lyra-1 execution plan as a JSON artifact to disk.

        This plan details all parameters and the command required to execute
        the Lyra-1 generation, but does not perform the execution itself.
        It's useful for debugging, reproducibility, or deferred execution.

        Args:
            plan_path: Optional target JSON file path for the plan. If None, it defaults to `output_root/lyra1_plan.json`.
            mode: Static or dynamic generation mode.
            output_root: Planned output directory where results would be saved if the plan were executed.
            prepared_input_path: Input path or a placeholder metadata string (e.g., "<image_input>") if input was not materialized.
            command: The full command-line arguments list generated for subprocess execution.
            prompt: Optional text prompt.
            trajectory: Camera trajectory name.
            checkpoint_dir: External checkpoint directory override.
            extra: Extra caller-provided metadata to include in the plan.

        Returns:
            A dictionary summarizing the generated plan, including its status and path.
        """
        # Determine the target path for the JSON plan.
        target = Path(plan_path) if plan_path is not None else output_root / "lyra1_plan.json"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory for the plan exists.

        # Construct the payload for the JSON plan.
        payload = {
            "status": "blocked",  # Indicates that the plan is created but not executed.
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "backend": "worldfoundry.lyra1.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self._runtime_root()),
            "checkpoint_dir": str(Path(checkpoint_dir or self.checkpoint_dir or "checkpoints").expanduser()),
            "command": command,
            "mode": mode,
            "prompt": prompt,
            "trajectory": trajectory,
            "output_root": str(output_root),
            "input_path": prepared_input_path,
            "blocked_reasons": list(self.BLOCKED_REASONS),  # Explain why execution is blocked by default.
            "extra": {key: str(value) for key, value in extra.items()},  # Convert extra metadata to strings.
        }

        # Write the JSON payload to the specified target file.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Return a summary of the plan for the caller.
        return {
            "status": "blocked",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "blocked_reasons": payload["blocked_reasons"],
        }

    def _prepare_input(
        self,
        visual_input: Any,
        mode: str,
        output_root: Path,
        fps: int,
        vipe_path: Optional[str] = None,
        materialize: bool = False,
    ) -> str:
        """
        Resolve or materialize a Lyra-1 input (image or video) to a file path.

        If `visual_input` is already a path, it's resolved. Otherwise, if `materialize` is True,
        the input (e.g., a PIL Image or list of frames) is saved to a file under `output_root`.

        Args:
            visual_input: Static image (Path, PIL.Image.Image, numpy.ndarray) or dynamic video input (Path, list of frames).
                          Can also be a string path.
            mode: Lyra-1 generation mode ("static" for image, "dynamic" for video).
            output_root: Output directory, used as a base for materializing non-path inputs.
            fps: Video materialization frame rate (only relevant for video inputs).
            vipe_path: Optional VIPE input path, takes precedence for dynamic mode if provided.
            materialize: Whether non-path inputs should be serialized to disk. If False, a placeholder string is returned.

        Returns:
            A string representing the absolute path to the prepared input file, or a placeholder string
            if the input was not a path and `materialize` was False.
        """
        if mode == "static":
            # Handle static image input.
            if isinstance(visual_input, (str, Path)):
                candidate = Path(visual_input).expanduser()
                # Return resolved path if it exists, otherwise the expanded path.
                return str(candidate.resolve()) if candidate.exists() else str(candidate)
            if materialize:
                # If not a path and materialize is True, save the image to disk.
                from worldfoundry.pipelines.lyra.lyra_utils import materialize_image_input

                return materialize_image_input(visual_input, str(output_root / "_inputs"))
            # If not a path and materialize is False, return a placeholder.
            return "<image_input>"

        # Handle dynamic video input.
        if vipe_path is not None:
            # If VIPE path is provided, prioritize it.
            candidate = Path(vipe_path).expanduser()
            return str(candidate.resolve()) if candidate.exists() else str(candidate)
        if isinstance(visual_input, (str, Path)):
            # If visual_input is a path, resolve it.
            candidate = Path(visual_input).expanduser()
            return str(candidate.resolve()) if candidate.exists() else str(candidate)
        if materialize:
            # If not a path and materialize is True, save the video to disk.
            from worldfoundry.pipelines.lyra.lyra_utils import materialize_video_input

            return materialize_video_input(visual_input, output_dir=str(output_root / "_inputs"), fps=fps)
        # If not a path and materialize is False, return a placeholder.
        return "<video_input>"

    def _locate_generated_video(self, output_root: Path, trajectory: str, multi_trajectory: bool) -> str:
        """
        Locate the primary generated Lyra-1 MP4 video file within the output directory.

        It searches in specific subdirectories based on `multi_trajectory` and `trajectory`
        before falling back to a recursive search.

        Args:
            output_root: The root directory where Lyra-1 stores its outputs.
            trajectory: The requested camera trajectory name. Used to prioritize search if `multi_trajectory` is True.
            multi_trajectory: Whether the generation was performed with `multi_trajectory` enabled,
                              which results in numbered subdirectories.

        Returns:
            A string representing the absolute path to the located MP4 video file.

        Raises:
            FileNotFoundError: If no MP4 video file is found within the specified `output_root`.
        """
        search_roots = []
        if multi_trajectory:
            # If multi_trajectory was enabled, Lyra-1 typically organizes outputs into numbered folders.
            preferred_index = self.MULTI_TRAJECTORY_INDEX.get(trajectory)
            if preferred_index is not None:
                # Prioritize the specific trajectory requested by the user.
                search_roots.append(output_root / str(preferred_index) / "rgb")
            # Also search all other numbered trajectory folders.
            search_roots.extend(sorted(output_root.glob("*/rgb")))
        else:
            # For single trajectory, the video is typically directly under `rgb` subfolder.
            search_roots.append(output_root / "rgb")

        # Iterate through potential root directories to find an MP4 file.
        for root in search_roots:
            if root.is_dir():
                mp4_files = sorted(root.glob("*.mp4"))
                if mp4_files:
                    return str(mp4_files[0].resolve())

        # As a fallback, perform a recursive search for any MP4 file within the output_root.
        fallback = sorted(output_root.rglob("*.mp4"))
        if fallback:
            return str(fallback[0].resolve())

        # If no video is found after all attempts, raise an error.
        raise FileNotFoundError(f"No generated Lyra-1 video found under: {output_root}")


__all__ = ["Lyra1Runtime"]