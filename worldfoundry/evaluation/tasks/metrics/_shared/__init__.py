"""Cross-metric helpers used by multiple evaluation modules.

Keep modules here only when at least one metric package imports them:

- ``clip_embed`` — CLIP encode/cosine helpers for ``semsr``, ``manipulation_direction``, ``quality_loss``
- ``bbox`` — IoU helpers for ``object_wise_consistency``
- ``perceptual`` — tensor/device helpers and ``compute_perceptual_bundle`` for ``lpips``, ``ssim``, ``ms_ssim``, ``psnr``, ``fsim``
- ``torch_fidelity`` — lazy loader for vendored ``vendor/torch_fidelity`` (``fid``, ``kid``, ``inception_score``, ``precision_recall``, ``ppl``, ``mind``)

Distribution helpers live in metric-local ``*/compute.py`` modules (for example ``fid/compute.py``), not here.
"""

from __future__ import annotations

__all__: list[str] = []
