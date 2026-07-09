from worldfoundry.base_models.diffusion_model.video.hunyuan_video.modules.loader import (
    load_model_from_registry,
)


def __getattr__(name):
    if name in {"HYVideoDiffusionTransformer", "HUNYUAN_VIDEO_CONFIG"}:
        from .models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer

        value = {
            "HYVideoDiffusionTransformer": HYVideoDiffusionTransformer,
            "HUNYUAN_VIDEO_CONFIG": HUNYUAN_VIDEO_CONFIG,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(name)


def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG

    return load_model_from_registry(
        args,
        in_channels,
        out_channels,
        factor_kwargs,
        HYVideoDiffusionTransformer,
        HUNYUAN_VIDEO_CONFIG,
    )
