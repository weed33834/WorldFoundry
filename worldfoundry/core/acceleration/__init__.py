"""Composable, in-tree diffusion acceleration techniques.

These utilities operate above individual kernels. They intentionally expose
the approximation policy so callers can validate quality for each model.
"""

from worldfoundry.core.acceleration.cache import AdaptiveResidualCache, FixedStepCache
from worldfoundry.core.acceleration.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4,
    quantize_nvfp4,
    replace_linear_with_nvfp4,
)
from worldfoundry.core.acceleration.quantization import (
    Float8Linear,
    replace_linear_with_float8,
    set_low_precision_enabled,
)
from worldfoundry.core.acceleration.technology import (
    AccelerationTechnology,
    acceleration_technology_report,
)
from worldfoundry.core.acceleration.token_pruning import (
    TokenPruner,
    TokenPruneState,
    prune_tokens,
    restore_tokens,
    select_token_indices,
)

__all__ = [
    "AdaptiveResidualCache",
    "AccelerationTechnology",
    "FixedStepCache",
    "Float8Linear",
    "NVFP4Linear",
    "TokenPruneState",
    "TokenPruner",
    "prune_tokens",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "replace_linear_with_float8",
    "replace_linear_with_nvfp4",
    "restore_tokens",
    "select_token_indices",
    "set_low_precision_enabled",
    "acceleration_technology_report",
]
