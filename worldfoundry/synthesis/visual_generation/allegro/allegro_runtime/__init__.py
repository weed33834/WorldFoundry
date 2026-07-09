from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


ALLEGRO_RUNTIME_UNAVAILABLE = (
    "Allegro source code is not available from the in-tree runtime package. "
    "WorldFoundry expects the upstream Allegro inference modules vendored under "
    "worldfoundry.synthesis.visual_generation.allegro.allegro_runtime/allegro."
)


@dataclass(frozen=True)
class AllegroComponents:
    transformer_cls: object
    autoencoder_cls: object
    to_tensor_video_cls: object
    center_crop_resize_video_cls: object
    pipeline_cls: object


def load_allegro_components() -> AllegroComponents:
    """
    Load the in-tree Allegro TI2V implementation components.

    Args:
        None: This loader has no parameters because checkpoints stay external assets.
    """
    runtime_root = Path(__file__).resolve().parent
    package_root = runtime_root / "allegro"
    if not package_root.exists():
        raise RuntimeError(ALLEGRO_RUNTIME_UNAVAILABLE)

    runtime_root_str = str(runtime_root)
    if runtime_root_str not in sys.path:
        sys.path.insert(0, runtime_root_str)

    try:
        from allegro.models.transformers.transformer_3d_allegro_ti2v import (
            AllegroTransformerTI2V3DModel,
        )
        from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
        from allegro.pipelines.data_process import (
            CenterCropResizeVideo,
            ToTensorVideo,
        )
        from allegro.pipelines.pipeline_allegro_ti2v import AllegroTI2VPipeline
    except ImportError as exc:
        raise RuntimeError(ALLEGRO_RUNTIME_UNAVAILABLE) from exc

    return AllegroComponents(
        transformer_cls=AllegroTransformerTI2V3DModel,
        autoencoder_cls=AllegroAutoencoderKL3D,
        to_tensor_video_cls=ToTensorVideo,
        center_crop_resize_video_cls=CenterCropResizeVideo,
        pipeline_cls=AllegroTI2VPipeline,
    )
