"""Module for base_models -> three_dimensions -> general_3d -> four_d_gaussians -> four_d_gaussians_runtime -> arguments -> hypernerf -> banana.py functionality."""

_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 250]
    },
)
OptimizationParams=dict(
)