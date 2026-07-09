from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


EASYANIMATE_RUNTIME_UNAVAILABLE = (
    "EasyAnimate source code is not available from the in-tree runtime package. "
    "WorldFoundry expects the upstream inference modules vendored under "
    "worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime/easyanimate."
)


@dataclass(frozen=True)
class EasyAnimateComponents:
    name_to_autoencoder_magvit: object
    name_to_transformer3d: object
    inpaint_pipeline_cls: object
    multi_text_encoder_inpaint_pipeline_cls: object
    convert_weight_dtype_wrapper: object
    get_image_to_video_latent: object


def load_easyanimate_components() -> EasyAnimateComponents:
    """
    Load the in-tree EasyAnimate implementation components.

    Args:
        None: This loader has no parameters because checkpoints stay external assets.
    """
    runtime_root = Path(__file__).resolve().parent
    package_root = runtime_root / "easyanimate"
    if not package_root.exists():
        raise RuntimeError(EASYANIMATE_RUNTIME_UNAVAILABLE)

    runtime_root_str = str(runtime_root)
    if runtime_root_str not in sys.path:
        sys.path.insert(0, runtime_root_str)

    try:
        from easyanimate.models import name_to_autoencoder_magvit, name_to_transformer3d
        from easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
        from easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper
        from easyanimate.utils.utils import get_image_to_video_latent
    except ImportError as exc:
        raise RuntimeError(EASYANIMATE_RUNTIME_UNAVAILABLE) from exc

    return EasyAnimateComponents(
        name_to_autoencoder_magvit=name_to_autoencoder_magvit,
        name_to_transformer3d=name_to_transformer3d,
        inpaint_pipeline_cls=EasyAnimateInpaintPipeline,
        multi_text_encoder_inpaint_pipeline_cls=EasyAnimateInpaintPipeline,
        convert_weight_dtype_wrapper=convert_weight_dtype_wrapper,
        get_image_to_video_latent=get_image_to_video_latent,
    )
