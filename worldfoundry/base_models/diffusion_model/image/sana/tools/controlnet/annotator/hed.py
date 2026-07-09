"""HED annotator entry point used by Sana ControlNet.

The original Sana ControlNet scripts import ``tools.controlnet.annotator.hed``.
WorldFoundry keeps that in-tree import stable while delegating the detector
implementation and Hugging Face cache handling to ``controlnet_aux``.
"""

from __future__ import annotations

import os

import numpy as np
from controlnet_aux import HEDdetector as _AuxHEDdetector


class HEDdetector:
    """Small compatibility wrapper around ``controlnet_aux.HEDdetector``."""

    def __init__(
        self,
        pretrained_model_or_path: str | None = None,
        *,
        filename: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        model_ref = pretrained_model_or_path or os.environ.get("WORLDFOUNDRY_HED_ANNOTATOR_PATH")
        if not model_ref:
            model_ref = "lllyasviel/Annotators"
        self.detector = _AuxHEDdetector.from_pretrained(
            model_ref,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def to(self, device: str):
        """Move the underlying HED network to a device and return self."""
        self.detector.to(device)
        return self

    def __call__(self, input_image, *args, **kwargs):
        """Return a uint8 numpy edge map without resizing the caller's image."""
        image = np.asarray(input_image, dtype=np.uint8)
        detect_resolution = int(min(image.shape[:2]))
        kwargs.setdefault("detect_resolution", detect_resolution)
        kwargs.setdefault("image_resolution", detect_resolution)
        kwargs.setdefault("output_type", "np")
        return np.asarray(self.detector(image, *args, **kwargs), dtype=np.uint8)
