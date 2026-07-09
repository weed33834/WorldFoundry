"""
Module for defining the StepVideoT2VSynthesis class, an in-tree wrapper for Step-Video text-to-video generation.

This module provides a synthesis class that prepares an execution plan for the Step-Video model
using its vendored runtime code. It handles configuration, argument parsing, and generates a JSON
payload representing the `torchrun` command and metadata needed to execute the generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from worldfoundry.core.io.paths import package_module_root as package_root
from typing import Any, Mapping, Sequence

from ...base_synthesis import BaseSynthesis
from worldfoundry.core.io import write_json
from worldfoundry.evaluation.models.runtime.profiles import DEFAULT_SHARED_HFD_ROOT


DEFAULT_STEP_VIDEO_CKPT = DEFAULT_SHARED_HFD_ROOT / "stepfun-ai--stepvideo-t2v"


class StepVideoT2VSynthesis(BaseSynthesis):
    """
    In-tree Step-Video-T2V runtime plan wrapper with vendored model code.

    This class facilitates the preparation of an execution plan for the Step-Video
    text-to-video generation model, leveraging its integrated runtime. It configures
    model parameters and outputs a JSON plan for subsequent execution.
    """

    MODEL_ID = "step-video-t2v"
    DISPLAY_NAME = "Step-Video-T2V"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_dir: str | Path = DEFAULT_STEP_VIDEO_CKPT,
        parallel: int = 4,
        tensor_parallel_degree: int = 2,
        ulysses_degree: int = 2,
    ) -> None:
        """
        Initializes the StepVideoT2VSynthesis wrapper.

        Args:
            model_id: Identifier for the model. Defaults to `MODEL_ID`.
            device: The compute device to target (e.g., "cuda").
            checkpoint_dir: Path to the model checkpoint directory. Defaults to `DEFAULT_STEP_VIDEO_CKPT`.
            parallel: Number of parallel processes/GPUs to use for generation.
            tensor_parallel_degree: Degree of tensor parallelism for distributed inference.
            ulysses_degree: Degree of Ulysses parallelism for distributed inference.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "text_to_video"
        self.device = device
        self.checkpoint_dir = str(Path(checkpoint_dir).expanduser())
        self.parallel = int(parallel)
        self.tensor_parallel_degree = int(tensor_parallel_degree)
        self.ulysses_degree = int(ulysses_degree)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "StepVideoT2VSynthesis":
        """
        Builds a lazy Step-Video runtime plan wrapper from pretrained options.

        This factory method allows instantiation with a flexible set of parameters,
        prioritizing explicitly passed arguments, then `pretrained_model_path` (if a mapping),
        and finally `kwargs` or default values.

        Args:
            pretrained_model_path: Optional checkpoint directory path or a mapping
                                   containing runtime options.
            args: Unused compatibility argument for pipeline factories; will be ignored.
            device: Target torch device label (e.g., "cuda").
            model_id: Runtime profile id, defaults to `step-video-t2v`.
            **kwargs: Additional runtime options such as tensor parallel degree,
                      checkpoint directory, etc.

        Returns:
            An instance of `StepVideoT2VSynthesis` configured with the provided options.
        """
        # Remove unused compatibility argument
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a direct path (not a mapping), set it as the checkpoint_dir.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options, overriding existing values.
        options.update(kwargs)
        return cls(
            # Determine model_id with fallback: explicit option > profile_id > method model_id > class MODEL_ID
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            # Determine device with fallback: explicit method device > option device > default "cuda"
            device=str(device or options.get("device") or "cuda"),
            # Determine checkpoint_dir with fallback: explicit option > model_dir option > default DEFAULT_STEP_VIDEO_CKPT
            checkpoint_dir=options.get("checkpoint_dir") or options.get("model_dir") or DEFAULT_STEP_VIDEO_CKPT,
            # Determine parallel processes, defaulting to 4
            parallel=int(options.get("parallel", 4)),
            # Determine tensor parallel degree, with 'tp_degree' as an alias, defaulting to 2
            tensor_parallel_degree=int(options.get("tensor_parallel_degree", options.get("tp_degree", 2))),
            # Determine ulysses degree, defaulting to 2
            ulysses_degree=int(options.get("ulysses_degree", 2)),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """
        Retrieves the package root path for the vendored Step-Video runtime.

        Returns:
            A `Path` object pointing to the Step-Video runtime directory within the package.
        """
        return package_root("worldfoundry.base_models.diffusion_model.video.step_video.step_video_runtime")

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
        """
        Prepares a Step-Video in-tree runtime execution plan for text-to-video generation.

        This method generates a JSON file that encapsulates the necessary `torchrun`
        command and metadata for executing the Step-Video model. It does not perform
        the actual generation but outputs a plan.

        Args:
            prompt: Text prompt for video generation.
            images: Must be `None` as this model is text-to-video only.
            video: Must be `None` as this model is text-to-video only.
            interactions: Optional metadata to be preserved in the plan (e.g., user interactions).
            output_path: Target path for the output JSON plan file. If `None`, defaults
                         to "step_video_plan.json" in the current working directory.
            fps: Optional frames per second metadata to include in the plan.
            **kwargs: Additional runtime options or service URLs (e.g., `vae_url`, `caption_url`,
                      `infer_steps`, `cfg_scale`, `time_shift`).

        Returns:
            A dictionary summarizing the prepared execution plan, including its status,
            model ID, artifact kind, and the path to the generated plan file.

        Raises:
            ValueError: If `images` or `video` inputs are provided, as this is a text-to-video model.
        """
        # Validate that no image or video inputs are provided for T2V model
        if images is not None or video is not None:
            raise ValueError("Step-Video-T2V does not accept image or video inputs.")

        # Get the root directory for the vendored Step-Video runtime code
        runtime_root = self._runtime_root()
        # Add the runtime root to sys.path to enable importing the vendored 'stepvideo' module
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))
        # Dynamically import the stepvideo module to ensure it's available in the runtime environment
        import stepvideo  # noqa: F401 - Imported for path discovery, not direct use here

        # Determine the target path for the output JSON plan file
        target = Path(output_path) if output_path is not None else Path.cwd() / "step_video_plan.json"
        # Resolve the absolute path and expand any user directories (~/)
        target = target.expanduser().resolve()

        # Construct the `torchrun` command as a list of strings
        command = [
            "torchrun",
            "--nproc_per_node",
            str(kwargs.get("parallel", self.parallel)),  # Number of processes/GPUs per node
            "run_parallel.py",  # The main script to execute within the runtime
            "--model_dir",
            str(Path(kwargs.get("checkpoint_dir") or self.checkpoint_dir).expanduser()),  # Path to model checkpoints
            "--vae_url",
            str(kwargs.get("vae_url", "127.0.0.1")),  # VAE service URL
            "--caption_url",
            str(kwargs.get("caption_url", "127.0.0.1")),  # Caption service URL
            "--ulysses_degree",
            str(kwargs.get("ulysses_degree", self.ulysses_degree)),  # Ulysses parallelism degree
            "--tensor_parallel_degree",
            str(kwargs.get("tensor_parallel_degree", self.tensor_parallel_degree)),  # Tensor parallelism degree
            "--prompt",
            prompt,  # Text prompt for generation
            "--infer_steps",
            str(kwargs.get("infer_steps", 50)),  # Number of inference steps
            "--cfg_scale",
            str(kwargs.get("cfg_scale", 9.0)),  # Classifier-free guidance scale
            "--time_shift",
            str(kwargs.get("time_shift", 13.0)),  # Time shift parameter for sampling
        ]

        # Construct the payload dictionary representing the structured execution plan
        payload = {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "backend": "worldfoundry.step_video.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self._runtime_root()),
            "command": command,
            "prompt": prompt,
            "interactions": list(interactions),
            "fps": fps,
            "note": (
                "Step-Video source is vendored in-tree. Full generation requires multi-GPU execution, "
                "caption/VAE API services, and staged Step-Video checkpoints."
            ),
        }
        # Write the execution plan payload to the specified JSON file
        write_json(target, payload)

        # Return a summary of the prepared plan
        return {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
        }