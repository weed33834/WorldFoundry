import importlib.util

from worldfoundry.base_models.diffusion_model.video.wan.variants.video_x_fun import (
    transformer as wan_transformer3d,
)
from worldfoundry.base_models.diffusion_model.video.wan.variants.video_x_fun.transformer import (
    WanRMSNorm,
    WanSelfAttention,
    WanTransformer3DModel,
)
from .transformer import VerseCrafterWanTransformer3DModel

__all__ = ["VerseCrafterWanTransformer3DModel"]

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   The simple_wrapper is used to solve the problem
    #   about conflicts between cython and torch.compile
    # --------------------------------------------------------------- #
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import wan_sparse_attention_wrapper

    WanSelfAttention.forward = simple_wrapper(wan_sparse_attention_wrapper()(WanSelfAttention.forward))
    print("Import Sparse Attention")

    WanTransformer3DModel.forward = simple_wrapper(WanTransformer3DModel.forward)

    # --------------------------------------------------------------- #
    #   CFG Skip Turbo
    # --------------------------------------------------------------- #
    if importlib.util.find_spec("paifuser.accelerator") is not None:
        from paifuser.accelerator import (cfg_skip_turbo, disable_cfg_skip,
                                          enable_cfg_skip, share_cfg_skip)
    else:
        from paifuser import (cfg_skip_turbo, disable_cfg_skip,
                              enable_cfg_skip, share_cfg_skip)

    WanTransformer3DModel.enable_cfg_skip = enable_cfg_skip()(WanTransformer3DModel.enable_cfg_skip)
    WanTransformer3DModel.disable_cfg_skip = disable_cfg_skip()(WanTransformer3DModel.disable_cfg_skip)
    WanTransformer3DModel.share_cfg_skip = share_cfg_skip()(WanTransformer3DModel.share_cfg_skip)

    print("Import CFG Skip Turbo")

    # --------------------------------------------------------------- #
    #   RMS Norm Kernel
    # --------------------------------------------------------------- #
    from paifuser.ops import rms_norm_forward
    WanRMSNorm.forward = rms_norm_forward
    print("Import PAI RMS Fuse")

    # --------------------------------------------------------------- #
    #   Fast Rope Kernel
    # --------------------------------------------------------------- #
    from paifuser.ops import (ENABLE_KERNEL, fast_rope_apply_qk,
                              rope_apply_real_qk)

    if ENABLE_KERNEL:
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            return fast_rope_apply_qk(q, k, grid_sizes, freqs)
    else:
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            return rope_apply_real_qk(q, k, grid_sizes, freqs)

    wan_transformer3d.rope_apply_qk = adaptive_fast_rope_apply_qk
    rope_apply_qk = adaptive_fast_rope_apply_qk
    print("Import PAI Fast rope")
