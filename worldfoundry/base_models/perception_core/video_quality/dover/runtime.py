"""Shared DOVER technical-quality inference used by PAI-Bench-C."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .paths import checkpoint_path as default_checkpoint_path
from .paths import config_path as default_config_path


class DOVERTechnicalScorer:
    """Technical-only DOVER view and score using the in-tree DOVER backbone."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        config: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
        self.config = Path(config) if config is not None else default_config_path()
        self.requested_device = device
        self.device: str | None = None
        self._model: Any = None
        self._sample_types: dict[str, Any] | None = None
        self._samplers: dict[str, Any] | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        import yaml

        from .dover.datasets.dover_datasets import UnifiedFrameSampler
        from .dover.models.evaluator import DOVER

        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"DOVER checkpoint not found: {self.checkpoint}; set WORLDFOUNDRY_DOVER_CKPT")
        options = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        model_args = dict(options["model"]["args"])
        model_args["backbone"] = {"technical": dict(model_args["backbone"]["technical"])}
        model_args["backbone_preserve_keys"] = "technical"
        self.device = "cuda" if self.requested_device == "auto" and torch.cuda.is_available() else self.requested_device
        if self.device == "auto":
            self.device = "cpu"
        self._model = DOVER(**model_args).to(self.device)
        state = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state, strict=False)
        self._model.eval()
        sample = dict(options["data"]["val-l1080p"]["args"]["sample_types"]["technical"])
        self._sample_types = {"technical": sample}
        self._samplers = {
            "technical": UnifiedFrameSampler(
                sample["clip_len"],
                sample.get("t_frag", sample["num_clips"]),
                sample["frame_interval"],
            )
        }
        return self._model

    def __call__(self, video_path: str | Path) -> float:
        import torch

        from .dover.datasets.dover_datasets import spatial_temporal_view_decomposition

        model = self._load()
        torch_state = torch.random.get_rng_state()
        numpy_state = np.random.get_state()
        random_state = random.getstate()
        try:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            views, _ = spatial_temporal_view_decomposition(str(video_path), self._sample_types, self._samplers)
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(random_state)
        mean = torch.tensor([123.675, 116.28, 103.53])
        std = torch.tensor([58.395, 57.12, 57.375])
        video = ((views["technical"].permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        clips = int(self._sample_types["technical"]["num_clips"])
        channels, frames, height, width = video.shape
        batch = video.reshape(channels, clips, frames // clips, height, width).permute(1, 0, 2, 3, 4).to(self.device)
        with torch.inference_mode():
            raw = model({"technical": batch}, reduce_scores=False)[0]
        raw_score = float(raw.detach().float().mean().cpu())
        return float(torch.sigmoid(torch.tensor((raw_score - 0.1107) / 0.07355)).item() * 100.0)
