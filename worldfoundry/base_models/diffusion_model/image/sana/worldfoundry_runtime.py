"""Module for base_models -> diffusion_model -> image -> sana -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import checkpoint_root_path, resolve_local_hf_model_path
from worldfoundry.base_models.diffusion_model.video.wan import wan_variant_root

from .variants import SANA_VARIANTS, config_root, get_sana_variant, runtime_root

_SANA_PACKAGE_DIR = Path(__file__).resolve().parent
SANA_WM_RUNTIME_DIR = _SANA_PACKAGE_DIR
SANA_WM_OFFICIAL_ENTRYPOINT = _SANA_PACKAGE_DIR / "inference_video_scripts" / "inference_sana_wm.py"
SANA_STREAMING_OFFICIAL_ENTRYPOINT = (
    _SANA_PACKAGE_DIR / "inference_video_scripts" / "v2v" / "inference_sana_streaming.py"
)


class SanaRuntime:
    """WorldFoundry adapter around the vendored official Sana inference code."""

    MODEL_ID = "sana"
    DISPLAY_NAME = "Sana"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cuda",
        python_executable: str | Path | None = None,
        default_work_dir: str | Path | None = None,
    ) -> None:
        """Init.

        Returns:
            The return value.
        """
        self.variant = get_sana_variant(model_id)
        self.model_id = self.variant.model_id
        self.model_name = self.variant.display_name
        self.generation_type = self.variant.task
        if model_path is None and self.variant.model_id == "sana-1600m-1024px-bf16":
            env_model_path = os.getenv("SANA_1600M_1024_BF16_PATH")
            if env_model_path and Path(env_model_path).is_file():
                self.model_path = str(env_model_path)
            else:
                self.model_path = str(self.variant.model_path)
        else:
            self.model_path = str(model_path or self.variant.model_path)
        self.config_path = str(config_path or self.variant.config_path)
        self.device = device
        self.python_executable = str(python_executable or sys.executable)
        self.default_work_dir = (
            Path(default_work_dir).expanduser()
            if default_work_dir is not None
            else self._project_root() / "tmp" / "sana"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "SanaRuntime":
        """Create a lazy Sana synthesis wrapper without importing the heavy runtime."""
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["model_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(
            options.get("model_id")
            or options.get("variant")
            or options.get("profile_id")
            or model_id
            or cls.MODEL_ID
        )
        return cls(
            model_id=resolved_model_id,
            model_path=options.get("model_path") or options.get("checkpoint_path"),
            config_path=options.get("config_path") or options.get("config"),
            device=str(device or options.get("device") or "cuda"),
            python_executable=options.get("python_executable") or options.get("python"),
            default_work_dir=options.get("work_dir") or options.get("default_work_dir"),
        )

    @staticmethod
    def available_variants() -> tuple[str, ...]:
        """Available variants.

        Returns:
            The return value.
        """
        return tuple(sorted(SANA_VARIANTS))

    @staticmethod
    def _project_root() -> Path:
        """Helper function to project root.

        Returns:
            The return value.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").is_file():
                return parent
        return current.parents[5]

    @staticmethod
    def _runtime_root() -> Path:
        """Helper function to runtime root.

        Returns:
            The return value.
        """
        return runtime_root().resolve()

    @staticmethod
    def _config_root() -> Path:
        """Helper function to config root.

        Returns:
            The return value.
        """
        return config_root().resolve()

    @staticmethod
    def _repo_src_root() -> Path:
        """Helper function to repo src root.

        Returns:
            The return value.
        """
        return Path(__file__).resolve().parents[5]

    @staticmethod
    def _wan_base_root() -> Path:
        """Helper function to wan base root.

        Returns:
            The return value.
        """
        return wan_variant_root("sana")

    def _subprocess_env(self) -> dict[str, str]:
        """Helper function to subprocess env.

        Returns:
            The return value.
        """
        root = self._runtime_root()
        src_root = self._repo_src_root()
        pythonpath_parts = [str(self._wan_base_root()), str(root), str(src_root)]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            pythonpath_parts.append(existing)
        env = os.environ.copy()
        for name in (
            "WORLDFOUNDRY_HF_LOCAL_FILES_ONLY",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "DIFFUSERS_OFFLINE",
        ):
            value = env.get(name)
            if value is not None and value.strip().lower() not in {"1", "true", "yes", "on"}:
                raise ValueError(
                    f"Sana inference requires offline model loading; {name}={value!r} is not allowed."
                )
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        env["WORLDFOUNDRY_SANA_CONFIG_ROOT"] = str(self._config_root())
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["WORLDFOUNDRY_HF_LOCAL_FILES_ONLY"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"
        return env

    def _resolve_config_path(self, value: str | Path) -> Path:
        """Helper function to resolve config path.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self._config_root() / path).resolve()

    @staticmethod
    def _local_hf_file(value: str) -> Path | None:
        """Helper function to local hf file.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        if not value.startswith("hf://"):
            return None
        parts = value[len("hf://") :].split("/")
        if len(parts) < 3:
            return None
        repo_dir = parts[1]
        relative = parts[2:]
        candidates = (
            checkpoint_root_path(repo_dir, *relative),
            checkpoint_root_path("hfd", f"{parts[0]}--{repo_dir}", *relative),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    @classmethod
    def _localize_config_value(cls, value: Any) -> tuple[Any, bool]:
        """Helper function to localize config value.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        if isinstance(value, str):
            local_file = cls._local_hf_file(value)
            if local_file is not None:
                return str(local_file), True
            if value.startswith("hf://"):
                raise FileNotFoundError(
                    f"Sana local asset is missing for {value!r}. "
                    "Pre-download the pinned repository before inference."
                )
            direct_checkpoint = checkpoint_root_path(value)
            if direct_checkpoint.is_dir():
                return str(direct_checkpoint.resolve()), True
            is_repo_id = (
                value.count("/") == 1
                and not value.startswith(("/", ".", "~"))
                and "://" not in value
            )
            if is_repo_id:
                try:
                    return str(resolve_local_hf_model_path(value)), True
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Sana checkpoint repository {value!r} is not staged locally. "
                        "Pre-download the pinned repository before inference."
                    ) from exc
            return value, False
        if isinstance(value, list):
            changed = False
            rows = []
            for item in value:
                localized, item_changed = cls._localize_config_value(item)
                rows.append(localized)
                changed = changed or item_changed
            return rows, changed
        if isinstance(value, dict):
            changed = False
            rows = {}
            for key, item in value.items():
                localized, item_changed = cls._localize_config_value(item)
                rows[key] = localized
                changed = changed or item_changed
            return rows, changed
        return value, False

    def _materialize_config(self, work_dir: Path) -> Path:
        """Helper function to materialize config.

        Args:
            work_dir: The work dir.

        Returns:
            The return value.
        """
        config_path = self._resolve_config_path(self.config_path)
        try:
            import yaml
        except Exception:
            return config_path
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            return config_path
        localized, changed = self._localize_config_value(config)
        if not changed:
            return config_path
        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / config_path.name
        target.write_text(yaml.safe_dump(localized, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return target.resolve()

    def _prepare_work_dir(self, output_path: str | Path | None) -> Path:
        """Helper function to prepare work dir.

        Args:
            output_path: The output path.

        Returns:
            The return value.
        """
        if output_path is not None:
            base = Path(output_path).expanduser()
            if base.suffix:
                return (base.parent / f".{base.stem}_sana_work").resolve()
            return (base / ".sana_work").resolve()
        return (self.default_work_dir / self.model_id).resolve()

    def _target_path(self, output_path: str | Path | None, suffix: str) -> Path:
        """Helper function to target path.

        Args:
            output_path: The output path.
            suffix: The suffix.

        Returns:
            The return value.
        """
        if output_path is None:
            target = self.default_work_dir / f"{self.model_id}{suffix}"
        else:
            target = Path(output_path).expanduser()
            if not target.suffix:
                target = target / f"{self.model_id}{suffix}"
            elif target.suffix.lower() != suffix.lower():
                target = target.with_suffix(suffix)
        return target.resolve()

    def _diffusers_model_ref(self) -> str | None:
        """Return a local or Hub diffusers checkpoint reference for Sana video."""
        repo_id = self.variant.diffusers_repo_id
        if not repo_id:
            return None
        repo_leaf = repo_id.split("/")[-1]
        owner, name = repo_id.split("/", 1) if "/" in repo_id else ("", repo_id)
        candidates = [
            Path(self.model_path).expanduser() if self.model_path and not self.model_path.startswith("hf://") else None,
            checkpoint_root_path(repo_leaf),
            checkpoint_root_path(name),
            checkpoint_root_path(f"{repo_leaf}"),
            checkpoint_root_path("hfd", f"{owner}--{name}") if owner else None,
            checkpoint_root_path("hfd", "hub", f"models--{owner}--{name}") if owner else None,
            Path(os.environ.get("HF_HUB_CACHE", "")) / f"models--{owner}--{name}" if owner and os.environ.get("HF_HUB_CACHE") else None,
        ]
        for candidate in candidates:
            if candidate and (candidate / "model_index.json").is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"Sana diffusers checkpoint {repo_id!r} is not staged locally. "
            "Pre-download the pinned repository before inference."
        )

    def _plan_path(self, output_path: str | Path | None) -> Path:
        """Helper function to plan path.

        Args:
            output_path: The output path.

        Returns:
            The return value.
        """
        if output_path is None:
            target = self.default_work_dir / f"{self.model_id}.json"
        else:
            target = Path(output_path).expanduser()
            if target.suffix:
                target = target.with_suffix(".json")
            else:
                target = target / f"{self.model_id}.json"
        return target.resolve()

    def _write_prompt_file(self, prompt: str, work_dir: Path) -> Path:
        """Helper function to write prompt file.

        Args:
            prompt: The prompt.
            work_dir: The work dir.

        Returns:
            The return value.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = work_dir / "prompt.txt"
        prompt_file.write_text((prompt or "").strip() + "\n", encoding="utf-8")
        return prompt_file

    def _materialize_image(self, images: Any, work_dir: Path) -> str:
        """Helper function to materialize image.

        Args:
            images: The images.
            work_dir: The work dir.

        Returns:
            The return value.
        """
        if images is None:
            raise ValueError(f"{self.model_id} requires a reference/control image input.")
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            if not images:
                raise ValueError(f"{self.model_id} received an empty image sequence.")
            images = images[0]
        if isinstance(images, (str, os.PathLike)):
            path = Path(images).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Sana input image not found: {path}")
            return str(path.resolve())

        from PIL import Image
        import numpy as np

        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / "control_input.png"
        if isinstance(images, Image.Image):
            image = images.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(images)).convert("RGB")
        image.save(target)
        return str(target)

    def _materialize_video(self, video: Any) -> str:
        """Resolve a source video path accepted by the official V2V runner."""
        if isinstance(video, Mapping):
            video = video.get("video_path") or video.get("path") or video.get("uri")
        if video is None:
            raise ValueError(f"{self.model_id} requires a source video input.")
        if not isinstance(video, (str, os.PathLike)):
            raise TypeError(
                f"{self.model_id} expects video to be a local path or hf:// URI, got {type(video).__name__}."
            )
        value = str(video)
        if value.startswith("hf://"):
            local_file = self._local_hf_file(value)
            return str(local_file) if local_file is not None else value
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SANA-Streaming source video not found: {path}")
        return str(path.resolve())

    @staticmethod
    def _latest_artifact(work_dir: Path, suffixes: tuple[str, ...]) -> Path:
        """Helper function to latest artifact.

        Args:
            work_dir: The work dir.
            suffixes: The suffixes.

        Returns:
            The return value.
        """
        candidates = [
            path
            for suffix in suffixes
            for path in work_dir.rglob(f"*{suffix}")
            if path.is_file() and ".sana_work" not in path.parts
        ]
        if not candidates:
            raise FileNotFoundError(f"Sana runner did not produce any artifact under {work_dir}.")
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _copy_artifact(source: Path, target: Path, artifact_kind: str) -> Path:
        """Helper function to copy artifact.

        Args:
            source: The source.
            target: The target.
            artifact_kind: The artifact kind.

        Returns:
            The return value.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        if artifact_kind == "generated_image" and target.suffix.lower() not in {source.suffix.lower(), ""}:
            from PIL import Image

            with Image.open(source) as image:
                image.convert("RGB").save(target)
        else:
            shutil.copyfile(source, target)
        return target

    @staticmethod
    def _sha256(path: Path) -> str:
        """Helper function to sha256.

        Args:
            path: The path.

        Returns:
            The return value.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _run_subprocess(self, argv: list[str], work_dir: Path, target: Path) -> None:
        """Helper function to run subprocess.

        Args:
            argv: The argv.
            work_dir: The work dir.
            target: The target.

        Returns:
            The return value.
        """
        root = self._runtime_root()
        log_path = target.with_suffix(target.suffix + ".sana.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Write output as it arrives.  The former subprocess.run(..., PIPE)
        # implementation kept every line in memory and left a minutes-long
        # CPFS/model startup looking completely silent until process exit.
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            process = subprocess.Popen(
                argv,
                cwd=str(root),
                env=self._subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode != 0:
            raise RuntimeError(
                f"Sana runner for {self.model_id} failed with code {returncode}; see {log_path}"
            )
        del work_dir

    def _run_diffusers_video(
        self,
        *,
        prompt: str,
        target: Path,
        seed: int,
        cfg_scale: float,
        step: int | None,
        fps: int,
        num_frames: int | None,
        height: int | None,
        width: int | None,
        max_sequence_length: int | None,
    ) -> dict[str, Any]:
        """Run the official Hugging Face diffusers SanaVideo pipeline."""
        model_ref = self._diffusers_model_ref()
        if model_ref is None:
            raise RuntimeError(f"{self.model_id} has no diffusers checkpoint reference configured.")
        default_height, default_width = self._video_default_size()
        resolved_height = int(height or default_height)
        resolved_width = int(width or default_width)

        import torch
        from diffusers import SanaVideoPipeline
        from diffusers.utils import export_to_video
        import diffusers

        if (
            not hasattr(diffusers, "SanaVideoCausalTransformer3DModel")
            and hasattr(diffusers, "SanaVideoTransformer3DModel")
        ):
            diffusers.SanaVideoCausalTransformer3DModel = diffusers.SanaVideoTransformer3DModel

        requested_local_only = os.environ.get("WORLDFOUNDRY_HF_LOCAL_FILES_ONLY")
        if requested_local_only is not None and requested_local_only.strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            raise ValueError(
                "Sana inference does not allow WORLDFOUNDRY_HF_LOCAL_FILES_ONLY=false."
            )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        pipe = SanaVideoPipeline.from_pretrained(model_ref, **load_kwargs)
        pipe.to(self.device)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        generator_device = "cuda" if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        output = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=step or self.variant.default_steps or 50,
            guidance_scale=cfg_scale,
            height=resolved_height,
            width=resolved_width,
            frames=num_frames or 81,
            generator=generator,
            output_type="pil",
            max_sequence_length=max_sequence_length or 300,
        )
        frames = output.frames[0] if output.frames and isinstance(output.frames[0], list) else output.frames
        target.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(target), fps=fps)
        return {
            "runtime": "diffusers.SanaVideoPipeline",
            "model_ref": model_ref,
            "height": resolved_height,
            "width": resolved_width,
            "frames": num_frames or 81,
        }

    def _video_default_size(self) -> tuple[int, int]:
        """Return the official default video size for the selected Sana video variant."""
        if self.variant.default_height is not None and self.variant.default_width is not None:
            return self.variant.default_height, self.variant.default_width
        if self.variant.resolution == "720p":
            return 720, 1280
        return 480, 832

    def _common_cli_args(
        self,
        *,
        script: Path,
        config_path: Path,
        work_dir: Path,
        prompt_file: Path | None,
        seed: int,
        cfg_scale: float,
        step: int | None,
    ) -> list[str]:
        """Helper function to common cli args.

        Returns:
            The return value.
        """
        local_model_path = self._local_hf_file(self.model_path)
        if self.model_path.startswith("hf://") and local_model_path is None:
            raise FileNotFoundError(
                f"Sana checkpoint {self.model_path!r} is not staged locally. "
                "Pre-download the pinned repository before inference."
            )
        resolved_model_path = str(local_model_path) if local_model_path is not None else self.model_path
        argv = [
            self.python_executable,
            str(script),
            "--config",
            str(config_path),
            "--model_path",
            resolved_model_path,
            "--work_dir",
            str(work_dir),
            "--sample_nums",
            "1",
            "--bs",
            "1",
            "--seed",
            str(seed),
            "--cfg_scale",
            str(cfg_scale),
        ]
        if prompt_file is not None:
            argv.extend(["--txt_file", str(prompt_file)])
        if step is not None:
            argv.extend(["--step", str(step)])
        return argv

    def _image_command(
        self,
        *,
        config_path: Path,
        work_dir: Path,
        prompt_file: Path,
        seed: int,
        cfg_scale: float,
        step: int | None,
    ) -> list[str]:
        """Helper function to image command.

        Returns:
            The return value.
        """
        script_name = "inference_sana_sprint.py" if self.variant.runner == "sprint" else "inference.py"
        return self._common_cli_args(
            script=self._runtime_root() / "scripts" / script_name,
            config_path=config_path,
            work_dir=work_dir,
            prompt_file=prompt_file,
            seed=seed,
            cfg_scale=cfg_scale,
            step=step,
        )

    def _controlnet_command(
        self,
        *,
        config_path: Path,
        work_dir: Path,
        prompt: str,
        image_path: str,
        seed: int,
        cfg_scale: float,
        step: int | None,
        controlmap_path: str | None,
    ) -> list[str]:
        """Helper function to controlnet command.

        Returns:
            The return value.
        """
        json_file = work_dir / "controlnet_input.json"
        key = "ref_controlmap_path" if controlmap_path else "ref_image_path"
        value = controlmap_path or image_path
        json_file.write_text(
            json.dumps([{"prompt": prompt, key: value}], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        argv = self._common_cli_args(
            script=self._runtime_root() / "tools" / "controlnet" / "inference_controlnet.py",
            config_path=config_path,
            work_dir=work_dir,
            prompt_file=None,
            seed=seed,
            cfg_scale=cfg_scale,
            step=step,
        )
        argv.extend(["--json_file", str(json_file)])
        return argv

    def _video_command(
        self,
        *,
        config_path: Path,
        work_dir: Path,
        prompt_file: Path,
        seed: int,
        cfg_scale: float,
        step: int | None,
        fps: int,
        num_frames: int | None,
    ) -> list[str]:
        """Helper function to video command.

        Returns:
            The return value.
        """
        argv = self._common_cli_args(
            script=self._runtime_root() / "inference_video_scripts" / "inference_sana_video.py",
            config_path=config_path,
            work_dir=work_dir,
            prompt_file=prompt_file,
            seed=seed,
            cfg_scale=cfg_scale,
            step=step,
        )
        argv.extend(["--fps", str(fps)])
        if num_frames is not None:
            argv.extend(["--num_frames", str(num_frames)])
        return argv

    def _streaming_command(
        self,
        *,
        config_path: Path,
        work_dir: Path,
        prompt: str,
        video_path: str,
        seed: int,
        cfg_scale: float,
        step: int,
        fps: int,
        num_frames: int,
        height: int,
        width: int,
        negative_prompt: str | None,
        flow_shift: float | None,
        motion_score: int | None,
        num_cached_blocks: int,
        sink_token: bool,
    ) -> list[str]:
        """Build the official SANA-Streaming V2V CLI invocation."""
        local_model_path = self._local_hf_file(self.model_path)
        if self.model_path.startswith("hf://") and local_model_path is None:
            raise FileNotFoundError(
                f"SANA-Streaming checkpoint {self.model_path!r} is not staged locally. "
                "Pre-download the pinned repository before inference."
            )
        resolved_model_path = str(local_model_path) if local_model_path is not None else self.model_path
        argv = [
            self.python_executable,
            str(SANA_STREAMING_OFFICIAL_ENTRYPOINT),
            "--mode",
            str(self.variant.mode or "long_streaming"),
            "--config",
            str(config_path),
            "--model_path",
            resolved_model_path,
            "--prompt",
            prompt,
            "--video_path",
            video_path,
            "--output_dir",
            str(work_dir),
            "--output_name",
            "sana_streaming_output.mp4",
            "--num_frames",
            str(num_frames),
            "--height",
            str(height),
            "--width",
            str(width),
            "--fps",
            str(fps),
            "--step",
            str(step),
            "--cfg_scale",
            str(cfg_scale),
            "--seed",
            str(seed),
            "--num_cached_blocks",
            str(num_cached_blocks),
            "--sink_token",
            str(sink_token).lower(),
        ]
        if negative_prompt is not None:
            argv.extend(["--negative_prompt", negative_prompt])
        if flow_shift is not None:
            argv.extend(["--flow_shift", str(flow_shift)])
        if motion_score is not None:
            argv.extend(["--motion_score", str(motion_score)])
        return argv

    def _plan(
        self,
        *,
        prompt: str,
        output_path: str | Path | None,
        command: list[str] | None = None,
        source_video_path: str | None = None,
    ) -> dict[str, Any]:
        """Helper function to plan.

        Returns:
            The return value.
        """
        payload = {
            "status": "planned",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "artifact_kind": self.variant.artifact_kind,
            "backend_quality": "execution_plan",
            "runtime_root": str(self._runtime_root()),
            "config_root": str(self._config_root()),
            "config_path": str(self._resolve_config_path(self.config_path)),
            "model_path": self.model_path,
            "prompt": prompt,
            "command": command,
            "notes": self.variant.notes,
        }
        if source_video_path is not None:
            payload["source_video_path"] = source_video_path
        if output_path is not None:
            target = self._plan_path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["artifact_path"] = str(target)
        return payload

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one Sana variant or return an explicit execution plan."""
        del interactions
        seed_value = kwargs.pop("seed", 42)
        seed = int(42 if seed_value is None else seed_value)
        cfg_scale = float(kwargs.pop("cfg_scale", kwargs.pop("guidance_scale", self.variant.default_cfg_scale or 4.5)))
        step = kwargs.pop("step", kwargs.pop("num_inference_steps", self.variant.default_steps))
        step = None if step is None else int(step)
        plan_only = bool(kwargs.pop("plan_only", False))
        num_frames_value = kwargs.pop("num_frames", kwargs.pop("frames", None))
        num_frames = None if num_frames_value is None else int(num_frames_value)
        height_value = kwargs.pop("height", None)
        width_value = kwargs.pop("width", None)
        height = None if height_value is None else int(height_value)
        width = None if width_value is None else int(width_value)
        max_sequence_length_value = kwargs.pop("max_sequence_length", None)
        max_sequence_length = None if max_sequence_length_value is None else int(max_sequence_length_value)
        work_dir = self._prepare_work_dir(output_path)
        target = self._target_path(output_path, self.variant.default_extension)
        prompt_file = self._write_prompt_file(prompt, work_dir)
        config_path = self._materialize_config(work_dir)
        source_video_path: str | None = None

        if self.variant.runner in {"image", "sprint"}:
            command = self._image_command(
                config_path=config_path,
                work_dir=work_dir,
                prompt_file=prompt_file,
                seed=seed,
                cfg_scale=cfg_scale,
                step=step,
            )
            suffixes = (".jpg", ".jpeg", ".png")
        elif self.variant.runner == "controlnet":
            controlmap_path = kwargs.pop("controlmap_path", None)
            image_path = self._materialize_image(images, work_dir)
            command = self._controlnet_command(
                config_path=config_path,
                work_dir=work_dir,
                prompt=prompt,
                image_path=image_path,
                seed=seed,
                cfg_scale=cfg_scale,
                step=step,
                controlmap_path=None if controlmap_path is None else str(controlmap_path),
            )
            suffixes = (".jpg", ".jpeg", ".png")
        elif self.variant.runner == "video":
            fps_value = int(fps or self.variant.default_fps or 16)
            diffusers_model_ref = self._diffusers_model_ref()
            if diffusers_model_ref is not None:
                default_height, default_width = self._video_default_size()
                command = [
                    "diffusers.SanaVideoPipeline.from_pretrained",
                    diffusers_model_ref,
                    f"frames={num_frames or 81}",
                    f"height={height or default_height}",
                    f"width={width or default_width}",
                    f"steps={step or self.variant.default_steps or 50}",
                    f"guidance_scale={cfg_scale}",
                    f"fps={fps_value}",
                ]
                if plan_only:
                    return self._plan(prompt=prompt, output_path=output_path, command=command)
                diffusers_metadata = self._run_diffusers_video(
                    prompt=prompt,
                    target=target,
                    seed=seed,
                    cfg_scale=cfg_scale,
                    step=step,
                    fps=fps_value,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    max_sequence_length=max_sequence_length,
                )
                return {
                    "status": "success",
                    "model_id": self.model_id,
                    "display_name": self.model_name,
                    "artifact_kind": self.variant.artifact_kind,
                    "artifact_path": str(target),
                    "artifact_sha256": self._sha256(target),
                    "generated_video_path": str(target),
                    "generated_image_path": None,
                    "backend_quality": "official_diffusers_runtime",
                    "runtime": diffusers_metadata["runtime"],
                    "runtime_root": str(self._runtime_root()),
                    "config_root": str(self._config_root()),
                    "config_path": str(config_path),
                    "source_config_path": str(self._resolve_config_path(self.config_path)),
                    "model_path": diffusers_metadata["model_ref"],
                    "prompt": prompt,
                    "video_metadata": diffusers_metadata,
                }
            command = self._video_command(
                config_path=config_path,
                work_dir=work_dir,
                prompt_file=prompt_file,
                seed=seed,
                cfg_scale=cfg_scale,
                step=step,
                fps=fps_value,
                num_frames=num_frames,
            )
            suffixes = (".mp4",)
        elif self.variant.runner == "streaming":
            source_video = kwargs.pop("video_path", kwargs.pop("source_video", video))
            source_video_path = self._materialize_video(source_video)
            fps_value = int(fps or self.variant.default_fps or 16)
            default_height, default_width = self._video_default_size()
            resolved_num_frames = int(num_frames or self.variant.default_num_frames or 969)
            resolved_height = int(height or default_height)
            resolved_width = int(width or default_width)
            command = self._streaming_command(
                config_path=config_path,
                work_dir=work_dir,
                prompt=prompt,
                video_path=source_video_path,
                seed=seed,
                cfg_scale=cfg_scale,
                step=int(step or self.variant.default_steps or 4),
                fps=fps_value,
                num_frames=resolved_num_frames,
                height=resolved_height,
                width=resolved_width,
                negative_prompt=kwargs.pop("negative_prompt", None),
                flow_shift=kwargs.pop("flow_shift", None),
                motion_score=kwargs.pop("motion_score", None),
                num_cached_blocks=int(kwargs.pop("num_cached_blocks", 2)),
                sink_token=bool(kwargs.pop("sink_token", True)),
            )
            suffixes = (".mp4",)
        else:
            raise ValueError(f"Unsupported Sana runner {self.variant.runner!r} for {self.model_id}.")

        if plan_only:
            return self._plan(
                prompt=prompt,
                output_path=output_path,
                command=command,
                source_video_path=source_video_path,
            )

        self._run_subprocess(command, work_dir, target)
        source = self._latest_artifact(work_dir, suffixes)
        target = self._copy_artifact(source, target, self.variant.artifact_kind)
        return {
            "status": "success",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "artifact_kind": self.variant.artifact_kind,
            "artifact_path": str(target),
            "artifact_sha256": self._sha256(target),
            "generated_video_path": str(target) if self.variant.artifact_kind == "generated_video" else None,
            "generated_image_path": str(target) if self.variant.artifact_kind == "generated_image" else None,
            "backend_quality": "official_in_tree_runtime",
            "runtime": "worldfoundry.synthesis.visual_generation.sana.official_cli",
            "runtime_root": str(self._runtime_root()),
            "config_root": str(self._config_root()),
            "config_path": str(config_path),
            "source_config_path": str(self._resolve_config_path(self.config_path)),
            "model_path": self.model_path,
            "prompt": prompt,
            "source_video_path": source_video_path,
        }


__all__ = [
    "SANA_STREAMING_OFFICIAL_ENTRYPOINT",
    "SANA_WM_OFFICIAL_ENTRYPOINT",
    "SANA_WM_RUNTIME_DIR",
    "SanaRuntime",
]
