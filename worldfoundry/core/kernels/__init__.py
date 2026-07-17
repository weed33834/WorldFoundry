"""In-tree accelerator kernels with portable PyTorch fallbacks."""

from worldfoundry.core.kernels.capabilities import KernelDeviceProfile, kernel_device_profile
from worldfoundry.core.kernels.diffusion import (
    clear_kernel_dispatch_cache,
    group_norm_silu,
    hidden_qk_rmsnorm_rope_3d,
    kernel_dispatch_report,
    layer_norm_scale_shift,
    qk_rmsnorm_rope,
    residual_gate_add,
    rms_norm_scale_shift,
    silu_and_mul,
    silu_mul,
)
from worldfoundry.core.kernels.moe import routed_swiglu_moe, routed_swiglu_moe_pytorch


def __getattr__(name: str):
    if name == "routed_swiglu_moe_triton":
        from worldfoundry.core.kernels.triton_moe import routed_swiglu_moe_triton

        return routed_swiglu_moe_triton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "clear_kernel_dispatch_cache",
    "group_norm_silu",
    "hidden_qk_rmsnorm_rope_3d",
    "KernelDeviceProfile",
    "kernel_device_profile",
    "kernel_dispatch_report",
    "layer_norm_scale_shift",
    "qk_rmsnorm_rope",
    "residual_gate_add",
    "rms_norm_scale_shift",
    "routed_swiglu_moe_pytorch",
    "routed_swiglu_moe",
    "routed_swiglu_moe_triton",
    "silu_and_mul",
    "silu_mul",
]
