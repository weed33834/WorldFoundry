"""Module for base_models -> diffusion_model -> video -> t2v_turbo -> t2v_turbo_runtime -> runtime.py functionality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from worldfoundry.core.io.paths import checkpoint_root_path


@dataclass(frozen=True)
class T2VTurboRuntimePlan:
    """V turbo runtime plan implementation."""
    runtime_root: Path
    required_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    config: Path
    model_ckpt: str
    lora_path: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable blocked-runtime plan with source and asset paths."""
        plan = asdict(self)
        plan["runtime_root"] = str(self.runtime_root)
        plan["config"] = str(self.config)
        plan["status"] = "blocked" if self.missing_files or self.missing_dependencies else "ready"
        plan["reason"] = self.reason
        plan["next_steps"] = (
            [
                "Vendor the listed source and config files under t2v_turbo_runtime.",
                "Rewrite vendored imports to package-relative imports.",
                "Keep model_ckpt and lora_path as external weight assets.",
            ]
            if self.missing_files or self.missing_dependencies
            else []
        )
        return plan

    @property
    def reason(self) -> str:
        """Describe the root blocker for initializing this runtime."""
        blockers = []
        if self.missing_files:
            blockers.append("missing in-tree T2V-Turbo source/config files")
        if self.missing_dependencies:
            blockers.append("missing Python runtime dependencies")
        if not blockers:
            return "T2V-Turbo runtime initialization plan is ready."
        return " and ".join(blockers)


class T2VTurboRuntimeBlockedError(RuntimeError):
    """V turbo runtime blocked error implementation."""
    def __init__(self, plan: T2VTurboRuntimePlan) -> None:
        """Describe why T2V-Turbo cannot initialize and expose the migration plan."""
        self.plan = plan
        missing_files = ", ".join(plan.missing_files)
        missing_dependencies = ", ".join(plan.missing_dependencies)
        details = "; ".join(
            detail
            for detail in (
                f"missing files: {missing_files}" if missing_files else "",
                f"missing dependencies: {missing_dependencies}" if missing_dependencies else "",
            )
            if detail
        )
        super().__init__(
            "T2V-Turbo runtime is blocked because its in-tree integration is incomplete "
            f"under {plan.runtime_root}: {details}"
        )


class T2VTurbo:
    """V turbo implementation."""
    def __init__(
        self,
        model_name: str,
        config: str,
        model_ckpt: str,
        lora_path: str,
        generation_type: Literal["t2v", "i2v"],
        num_frames: int = 48,
        fps: int = 16,
        seed: int = 0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 4,
        motion_gs: float = 0.05,
        use_motion_cond: bool = False,
        percentage: float = 0.3,
        lcm_origin_steps: int = 200,
        param_dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
        plan_only: bool = False,
    ) -> None:
        """Initialize the in-tree T2V-Turbo runtime.

        Args:
            model_name: Registry model identifier.
            config: External T2V-Turbo model config path.
            model_ckpt: External base VideoCrafter checkpoint path.
            lora_path: External T2V-Turbo LoRA checkpoint path.
            generation_type: Supported generation mode; only ``t2v`` is valid.
            num_frames: Number of frames to generate.
            fps: Output video frame rate.
            seed: Torch random seed.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Scheduler inference step count.
            motion_gs: Official T2V-Turbo v1 motion guidance scale.
            use_motion_cond: Enable the motion-condition path used by T2V-Turbo-v2 MG checkpoints.
            percentage: Fraction threshold used by the official motion guidance schedule.
            lcm_origin_steps: Scheduler origin step count from the official Gradio demo.
            param_dtype: Official Gradio dtype selection.
            plan_only: Return a structured blocked plan without loading runtime code.
        """
        if generation_type == "i2v":
            raise ValueError("T2V-Turbo runtime only supports text-to-video generation.")

        self.model_name = model_name
        self.fps = fps
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.motion_gs = motion_gs
        self.use_motion_cond = use_motion_cond
        self.percentage = percentage
        self.lcm_origin_steps = lcm_origin_steps
        self.param_dtype = param_dtype
        self.model_ckpt = self._resolve_model_ckpt(model_ckpt)
        self.lora_path = self._resolve_lora_path(lora_path)
        self.plan = self._runtime_plan(config, str(self.model_ckpt), str(self.lora_path))
        self.plan_only = plan_only

        if plan_only:
            return
        if self.plan.missing_files or self.plan.missing_dependencies:
            raise T2VTurboRuntimeBlockedError(self.plan)

        import torch
        from omegaconf import OmegaConf

        from .source.pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
        from .source.scheduler.t2v_turbo_scheduler import T2VTurboScheduler
        from .source.utils.common_utils import load_model_checkpoint
        from .source.utils.lora import collapse_lora, monkeypatch_remove_lora
        from .source.utils.lora_handler import LoraHandler
        from worldfoundry.base_models.diffusion_model.video.lvdm.utils import instantiate_from_config

        torch.manual_seed(seed)
        model_config = OmegaConf.load(str(self.plan.config)).pop("model", OmegaConf.create())
        pretrained_t2v = instantiate_from_config(model_config)
        pretrained_t2v = load_model_checkpoint(pretrained_t2v, str(self.model_ckpt))

        unet_config = model_config["params"]["unet_config"]
        unet_config["params"]["use_checkpoint"] = False
        unet_config["params"]["time_cond_proj_dim"] = 256
        unet = instantiate_from_config(unet_config)
        unet.load_state_dict(
            pretrained_t2v.model.diffusion_model.state_dict(), strict=False
        )

        use_unet_lora = True
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        lora_manager.add_lora_to_model(
            use_unet_lora,
            unet,
            lora_manager.unet_replace_modules,
            lora_path=str(self.lora_path),
            dropout=0.1,
            r=64,
        )
        unet.eval()
        collapse_lora(unet, lora_manager.unet_replace_modules)
        monkeypatch_remove_lora(unet)

        pretrained_t2v.model.diffusion_model = unet
        scheduler = T2VTurboScheduler(
            linear_start=model_config["params"]["linear_start"],
            linear_end=model_config["params"]["linear_end"],
        )
        self.pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
        self.pipeline.set_runtime_dtype(next(pretrained_t2v.parameters()).dtype)
        self.pipeline = self.pipeline.to(device="cuda")

    @staticmethod
    def _hf_uri_parts(value: str) -> tuple[str, str] | None:
        text = str(value or "").strip()
        if not text.startswith("hf://"):
            return None
        repo_and_file = text[len("hf://") :]
        parts = repo_and_file.split("/")
        if len(parts) < 3:
            raise ValueError(f"Expected hf://owner/repo/path, got {value!r}")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return repo_id, filename

    @staticmethod
    def _download_hf_file(repo_id: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo_id, filename=filename)).resolve()

    @classmethod
    def _resolve_asset(
        cls,
        value: str,
        *,
        repo_id: str,
        filename: str,
        local_candidates: tuple[Path, ...],
    ) -> Path:
        raw = str(value or "").strip()
        for candidate in local_candidates:
            if candidate.is_file():
                return candidate.resolve()
        hf_parts = cls._hf_uri_parts(raw)
        if hf_parts is not None:
            return cls._download_hf_file(*hf_parts)
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()
        return cls._download_hf_file(repo_id, filename)

    @classmethod
    def _resolve_model_ckpt(cls, value: str) -> Path:
        return cls._resolve_asset(
            value,
            repo_id="VideoCrafter/VideoCrafter2",
            filename="model.ckpt",
            local_candidates=(
                checkpoint_root_path("video_models", "videocrafter_t2v_512_v2.ckpt"),
                checkpoint_root_path("VideoCrafter", "model.ckpt"),
                checkpoint_root_path("hfd", "VideoCrafter--VideoCrafter2", "model.ckpt"),
            ),
        )

    @classmethod
    def _resolve_lora_path(cls, value: str) -> Path:
        return cls._resolve_asset(
            value,
            repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
            filename="unet_lora.pt",
            local_candidates=(
                checkpoint_root_path("video_models", "t2v_turbo_unet_lora.pt"),
                checkpoint_root_path("jiachenli-ucsb", "T2V-Turbo-VC2", "unet_lora.pt"),
                checkpoint_root_path("hfd", "jiachenli-ucsb--T2V-Turbo-VC2", "unet_lora.pt"),
            ),
        )

    @staticmethod
    def _runtime_plan(config: str, model_ckpt: str, lora_path: str) -> T2VTurboRuntimePlan:
        """Build the blocked-runtime plan from runtime source and external assets."""
        runtime_root = Path(__file__).resolve().parent
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = runtime_root / "configs" / config_path.name

        required_files = (
            "configs/inference_t2v_512_v2.0.yaml",
            "source/pipeline/t2v_turbo_vc2_pipeline.py",
            "source/scheduler/t2v_turbo_scheduler.py",
            "../../lvdm/modules/networks/openaimodel3d.py",
            "source/utils/lora.py",
            "source/utils/lora_handler.py",
            "source/utils/common_utils.py",
            "source/utils/utils.py",
        )
        missing_files = tuple(
            required_file
            for required_file in required_files
            if not (runtime_root / required_file).is_file()
        )
        missing_dependencies = tuple(
            dependency
            for dependency in ("torch", "omegaconf", "safetensors", "huggingface_hub")
            if find_spec(dependency) is None
        )
        return T2VTurboRuntimePlan(
            runtime_root=runtime_root,
            required_files=required_files,
            missing_files=missing_files,
            missing_dependencies=missing_dependencies,
            config=config_path,
            model_ckpt=model_ckpt,
            lora_path=lora_path,
        )

    def generate_video(self, prompt: str, image_path: str | None = None):
        """Generate video frames from a text prompt.

        Args:
            prompt: Text prompt used by T2V-Turbo.
            image_path: Unsupported image path; must be ``None`` for T2V.
        """
        if image_path is not None:
            raise ValueError("T2V-Turbo runtime does not accept an input image.")
        if self.plan_only:
            return self.plan.to_dict()

        import torch

        if self.param_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.param_dtype == "fp16":
            dtype = torch.float16
        elif self.param_dtype == "fp32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unknown T2V-Turbo param_dtype: {self.param_dtype}")

        self.pipeline.set_runtime_dtype(dtype)
        self.pipeline.unet.dtype = dtype
        self.pipeline.unet.to("cuda", dtype)
        self.pipeline.text_encoder.to("cuda", dtype)
        self.pipeline.vae.to("cuda", dtype)
        self.pipeline.to("cuda", dtype)

        video = self.pipeline(
            prompt=prompt,
            frames=self.num_frames,
            fps=self.fps,
            guidance_scale=self.guidance_scale,
            motion_gs=self.motion_gs,
            use_motion_cond=self.use_motion_cond,
            percentage=self.percentage,
            num_inference_steps=self.num_inference_steps,
            lcm_origin_steps=self.lcm_origin_steps,
            num_videos_per_prompt=1,
        )

        video = video.detach().squeeze().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = (video + 1.0) / 2.0
        video = video.permute(1, 0, 2, 3)
        return video
