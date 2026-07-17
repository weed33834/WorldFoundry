"""Reusable SAM2 box-prompt video tracking adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def stage_video_frames(
    frames: list[np.ndarray] | np.ndarray,
    output_dir: Path,
    *,
    size: tuple[int, int] | None = None,
    jpeg_quality: int = 95,
) -> list[np.ndarray]:
    """Materialize the numbered JPEG layout consumed by SAM2 video inference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        if size is not None and image.size != (size[1], size[0]):
            image = image.resize((size[1], size[0]), resample=Image.Resampling.BICUBIC)
        path = output_dir / f"{index:05d}.jpg"
        image.save(path, format="JPEG", quality=jpeg_quality, subsampling=0)
        staged.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    return staged


class SAM2MaskTracker:
    """Lazily load one in-tree SAM2 predictor and track integer object IDs."""

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam2.1-hiera-base-plus",
        checkpoint: Path | None = None,
        config_name: str | None = None,
        device: str = "auto",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
    ) -> None:
        self.model_id = model_id
        self.checkpoint = checkpoint.expanduser().resolve() if checkpoint is not None else None
        self.config_name = config_name
        self.requested_device = device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._predictor: Any = None
        self._resolved_checkpoint: Path | None = None
        self._resolved_config: str | None = None
        self._device: str | None = None

    def _resolve_device(self) -> str:
        import torch

        if self.requested_device != "auto":
            return self.requested_device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_model_files(self) -> tuple[str, Path]:
        from .build_sam import HF_MODEL_ID_TO_FILENAMES

        if self.model_id not in HF_MODEL_ID_TO_FILENAMES and self.config_name is None:
            known = ", ".join(sorted(HF_MODEL_ID_TO_FILENAMES))
            raise ValueError(f"unknown SAM2 model ID {self.model_id!r}; expected one of: {known}")
        default_config, default_filename = HF_MODEL_ID_TO_FILENAMES.get(self.model_id, (self.config_name, None))
        config_name = self.config_name or default_config
        if not config_name:
            raise ValueError("SAM2 config name is required")
        checkpoint = self.checkpoint
        if checkpoint is None:
            for variable in (
                "WORLDFOUNDRY_SAM2_CKPT",
                "WORLDFOUNDRY_WORLDBENCH_SAM2_CKPT",
                "WORLDFOUNDRY_PHYSICAL_AI_BENCH_SAM2_CKPT",
            ):
                value = os.environ.get(variable)
                if value:
                    checkpoint = Path(value).expanduser().resolve()
                    break
        if checkpoint is None and self.model_id == "facebook/sam2.1-hiera-base-plus":
            from . import checkpoint_path

            candidate = checkpoint_path()
            if candidate.is_file():
                checkpoint = candidate.resolve()
        if checkpoint is None:
            if not default_filename:
                raise FileNotFoundError("SAM2 checkpoint was not provided")
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError("huggingface-hub is required to resolve the SAM2 checkpoint") from exc
            checkpoint = Path(hf_hub_download(repo_id=self.model_id, filename=default_filename)).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        return str(config_name), checkpoint

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        from .build_sam import build_sam2_video_predictor

        config_name, checkpoint = self._resolve_model_files()
        device = self._resolve_device()
        self._predictor = build_sam2_video_predictor(config_name, str(checkpoint), device=device, mode="eval")
        self._resolved_config = config_name
        self._resolved_checkpoint = checkpoint
        self._device = device
        return self._predictor

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "implementation": "worldfoundry.base_models.perception_core.segment.sam2",
            "model_id": self.model_id,
            "config": self._resolved_config or self.config_name,
            "checkpoint": str(self._resolved_checkpoint) if self._resolved_checkpoint else None,
            "device": self._device or self.requested_device,
        }

    def track(
        self,
        frames_dir: Path,
        prompts: dict[int, tuple[int, np.ndarray]],
        *,
        frame_count: int,
        size: tuple[int, int],
    ) -> list[np.ndarray]:
        if not prompts:
            return [np.zeros(size, dtype=np.int32) for _ in range(frame_count)]
        predictor = self._load()
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
        )
        try:
            for object_id, (frame_index, box) in sorted(prompts.items()):
                if frame_index < frame_count:
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=int(frame_index),
                        obj_id=int(object_id),
                        box=np.asarray(box, dtype=np.float32),
                    )
            labels = [np.zeros(size, dtype=np.int32) for _ in range(frame_count)]

            def consume(iterator: Any) -> None:
                for frame_index, object_ids, mask_logits in iterator:
                    if not 0 <= int(frame_index) < frame_count:
                        continue
                    frame_labels = labels[int(frame_index)]
                    for mask_index, object_id in enumerate(object_ids):
                        mask = (mask_logits[mask_index] > 0.0).detach().cpu().numpy()
                        if mask.ndim == 3:
                            mask = mask[0]
                        if mask.shape != size:
                            image = Image.fromarray(mask.astype(np.uint8) * 255)
                            image = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
                            mask = np.asarray(image) > 0
                        frame_labels[mask.astype(bool)] = int(object_id)

            earliest = min(frame_index for frame_index, _ in prompts.values())
            consume(predictor.propagate_in_video(state, start_frame_idx=earliest))
            if earliest > 0:
                consume(predictor.propagate_in_video(state, start_frame_idx=earliest, reverse=True))
            return labels
        finally:
            predictor.reset_state(state)
