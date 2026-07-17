"""Module for base_models -> diffusion_model -> video -> skyreels_v3 -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_DIR.parents[4]
ENTRYPOINT = RUNTIME_DIR / "generate_video.py"
BLOCKED_REASON = (
    "SkyReels V3 official source is vendored in-tree; execution still requires "
    "the official dependency environment, checkpoints, and task assets."
)


def _resolve_existing_local_path(value: Any) -> str:
    """Return an absolute path for local inputs while preserving remote URLs/model IDs."""
    text = str(value)
    candidate = Path(text).expanduser()
    return str(candidate.resolve()) if candidate.exists() else text


class SkyReelsV3Runtime:
    """In-tree runtime shim for SkyReels V3."""

    SUPPORTED_TASKS = {
        "single_shot_extension",
        "shot_switching_extension",
        "reference_to_video",
        "talking_avatar",
    }

    def __init__(
        self,
        model_path: str | None,
        task_type: str,
        device: str,
        loaded: bool = False,
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Store SkyReels V3 runtime configuration.

        Args:
            model_path: Local checkpoint directory or package model id.
            task_type: SkyReels V3 task type.
            device: Runtime device label.
            loaded: Whether a complete in-tree engine is available.
            options: Loader options preserved for diagnostics.
        """
        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported SkyReels V3 task_type: {task_type}")
        self.model_path = model_path
        self.task_type = task_type
        self.device = device
        self.loaded = loaded
        self.options = dict(options or {})

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | None,
        task_type: str,
        device: str,
        **options: Any,
    ) -> "SkyReelsV3Runtime":
        """
        Build the runtime shim against the in-tree official source.

        Args:
            model_path: Local checkpoint directory or package model id.
            task_type: SkyReels V3 task type.
            device: Runtime device label.
            **options: Loader options preserved for diagnostics.
        """
        return cls(
            model_path=model_path,
            task_type=task_type,
            device=device,
            loaded=bool(options.pop("load_engine", False)),
            options=options,
        )

    def generate(self, task_type: str, **kwargs: Any) -> Any:
        """
        Run the in-tree official SkyReels V3 entrypoint when explicitly loaded.

        Args:
            task_type: Selected SkyReels V3 task type.
            **kwargs: Generation arguments accepted by the pipeline.
        """
        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported SkyReels V3 task_type: {task_type}")
        use_usp = _truthy_value(kwargs.get("use_usp"))
        if use_usp and _truthy_value(kwargs.get("low_vram")):
            raise ValueError("SkyReels V3 cannot use low_vram and use_usp together.")
        output_path = (
            Path(str(kwargs["output_path"])).expanduser().resolve()
            if kwargs.get("output_path")
            else None
        )
        nproc_per_node = _requested_nproc_per_node(kwargs) if use_usp else 0
        if use_usp and nproc_per_node <= 1:
            raise ValueError(
                "SkyReels V3 use_usp requires torchrun with at least two visible GPUs. "
                "Set CUDA_VISIBLE_DEVICES or pass nproc_per_node/torchrun_nproc_per_node."
            )
        entrypoint = [
            str(ENTRYPOINT),
            "--task_type",
            task_type,
            "--prompt",
            str(kwargs.get("prompt") or ""),
            "--duration",
            str(int(kwargs.get("duration") or 5)),
            "--seed",
            str(int(kwargs.get("seed") or 42)),
            "--resolution",
            str(kwargs.get("resolution") or "720P"),
        ]
        if self.model_path:
            entrypoint.extend(["--model_id", _resolve_existing_local_path(self.model_path)])
        if kwargs.get("video") is not None:
            entrypoint.extend(["--input_video", _resolve_existing_local_path(kwargs["video"])])
        if kwargs.get("audio") is not None:
            entrypoint.extend(["--input_audio", _resolve_existing_local_path(kwargs["audio"])])
        images = kwargs.get("images")
        if images is not None:
            if task_type == "talking_avatar":
                image_arg = images[0] if isinstance(images, (list, tuple)) else images
                entrypoint.extend(["--input_image", _resolve_existing_local_path(image_arg)])
            else:
                ref_imgs = (
                    ",".join(_resolve_existing_local_path(image) for image in images)
                    if isinstance(images, (list, tuple))
                    else _resolve_existing_local_path(images)
                )
                entrypoint.extend(["--ref_imgs", ref_imgs])
        if use_usp:
            entrypoint.append("--use_usp")
        if _truthy_value(kwargs.get("offload")):
            entrypoint.append("--offload")
        if _truthy_value(kwargs.get("low_vram")):
            entrypoint.append("--low_vram")
        if output_path is not None:
            entrypoint.extend(["--output_dir", str(output_path)])

        python_executable = str(kwargs.get("python_executable") or sys.executable)
        if use_usp:
            argv = [
                python_executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={nproc_per_node}",
                *entrypoint,
            ]
        else:
            argv = [python_executable, *entrypoint]

        env = os.environ.copy()
        # The vendored entrypoint imports both its local ``skyreels_v3`` package
        # and shared ``worldfoundry`` modules.  Because the subprocess runs with
        # ``RUNTIME_DIR`` as its cwd, the repository root is not otherwise on
        # sys.path when this checkout has not been installed as a wheel.
        pythonpath = [str(REPO_ROOT), str(RUNTIME_DIR), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
        started_at = time.time()
        completed = subprocess.run(
            argv,
            cwd=str(RUNTIME_DIR),
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = output_path.with_suffix(output_path.suffix + ".log")
            log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"SkyReels V3 official runtime failed with code {completed.returncode}: {completed.stdout}"
            )
        artifact_path = self._resolve_artifact(output_path, since=started_at)
        if artifact_path is None:
            return {
                "status": "completed",
                "stdout": completed.stdout,
                "runtime_dir": str(RUNTIME_DIR),
                "log_path": str(log_path) if log_path else None,
            }
        if output_path is not None and output_path.suffix.lower() == ".mp4" and artifact_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, output_path)
            artifact_path = output_path
        return {
            "status": "completed",
            "artifact_path": str(artifact_path),
            "output_path": str(artifact_path),
            "stdout": completed.stdout,
            "runtime_dir": str(RUNTIME_DIR),
            "log_path": str(log_path) if log_path else None,
        }

    def _resolve_artifact(self, output_path: Path | None, *, since: float) -> Path | None:
        """Helper function to resolve artifact.

        Args:
            output_path: The output path.

        Returns:
            The return value.
        """
        roots: list[Path] = []
        if output_path is not None:
            roots.append(output_path if output_path.suffix.lower() != ".mp4" else output_path.parent)
            if output_path.is_file() and output_path.stat().st_mtime >= since - 1.0:
                return output_path
        roots.append(RUNTIME_DIR / "result")
        candidates: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            candidates.extend(
                path
                for path in root.rglob("*.mp4")
                if path.is_file() and path.stat().st_mtime >= since - 1.0
            )
        if not candidates:
            return None
        return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.stat().st_size), reverse=True)[0]


def _truthy_value(value: Any) -> bool:
    """Helper function to truthy value.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _requested_nproc_per_node(kwargs: dict[str, Any]) -> int:
    """Helper function to requested nproc per node.

    Args:
        kwargs: The kwargs.

    Returns:
        The return value.
    """
    for key in ("nproc_per_node", "torchrun_nproc_per_node", "torchrun_nproc", "num_gpus", "world_size"):
        value = kwargs.get(key)
        if value in {None, ""}:
            continue
        try:
            nproc = int(value)
        except Exception:
            continue
        if nproc > 0:
            return nproc

    visible_count = _cuda_visible_device_count(os.getenv("CUDA_VISIBLE_DEVICES"))
    if visible_count > 0:
        return visible_count

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _cuda_visible_device_count(value: str | None) -> int:
    """Helper function to cuda visible device count.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    if not value:
        return 0
    return len([item for item in value.split(",") if item.strip()])


__all__ = ["BLOCKED_REASON", "ENTRYPOINT", "RUNTIME_DIR", "SkyReelsV3Runtime"]
