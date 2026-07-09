"""Module for base_models -> three_dimensions -> general_3d -> four_d_gaussians -> four_d_gaussians_runtime -> utils -> params_utils.py functionality."""

def merge_hparams(args, config):
    """Merge hparams.

    Args:
        args: The args.
        config: The config.
    """
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    return args