"""Module for base_models -> diffusion_model -> video -> hunyuan_video -> modules -> loader.py functionality."""

def load_model_from_registry(args, in_channels, out_channels, factor_kwargs, model_cls, config_map):
    """Load a Hunyuan video transformer from a model config registry."""
    if args.model not in config_map:
        raise NotImplementedError()
    return model_cls(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        **config_map[args.model],
        **factor_kwargs,
    )


__all__ = ["load_model_from_registry"]
