"""MoVerse Panorama Generation Module.

Converts perspective images to 360° equirectangular panoramas using:
- AutoLevel: single-image camera calibration (vFOV / pitch / roll)
- FLUX.1-Fill-dev + LoRA: inpainting/outpainting for panorama completion
"""

from .prompt_builder import build_pano_prompt, derive_video_prompt, TRIGGER_WORDS, FLUX_SUFFIX
