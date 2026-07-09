"""Module for base_models -> three_dimensions -> general_3d -> four_d_gaussians -> four_d_gaussians_runtime -> arguments -> dnerf -> lego.py functionality."""

_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 25]
    },

    # deformation_lr_init = 0.001,
    # deformation_lr_final = 0.001,
    # deformation_lr_delay_mult = 0.01,
    # grid_lr_init = 0.001,
    # grid_lr_final = 0.001,
)