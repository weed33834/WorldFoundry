"""Aesthetic quality metric — LAION aesthetic predictor on CLIP ViT-L/14 features."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception import openai_clip

from ..base import BaseMetric
from ..weight_utils import wbench_asset_path


class AestheticQualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        clip_path = str(wbench_asset_path("wbench_clip_vit_l14_checkpoint"))
        self.clip_model, self.preprocess = openai_clip.load(clip_path, device=self.device)
        self.aesthetic_model = self._get_aesthetic_model()

    @property
    def name(self):
        return "aesthetic_quality"

    def _get_aesthetic_model(self):
        path_to_model = wbench_asset_path("wbench_aesthetic_linear_checkpoint")
        model = nn.Linear(768, 1)
        state_dict = torch.load(path_to_model, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        scores = []
        for frame in frames:
            img = self.preprocess(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.encode_image(img).to(torch.float32)
                feats = F.normalize(feats, dim=-1, p=2)
                score = self.aesthetic_model(feats).item()
                scores.append(score / 10.0)
        return {f"{self.name}_score": float(np.mean(scores))}
