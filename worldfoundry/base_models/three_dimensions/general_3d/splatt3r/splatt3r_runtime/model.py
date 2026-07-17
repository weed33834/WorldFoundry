"""Module for base_models -> three_dimensions -> general_3d -> splatt3r -> splatt3r_runtime -> model.py functionality."""

import einops
import torch
from torch import nn

from worldfoundry.base_models.three_dimensions.general_3d.mast3r import ensure_import_paths

from .utils import geometry, sh_utils

ensure_import_paths()
import mast3r.model as mast3r_model


class MAST3RGaussians(nn.Module):
    """Inference-only Splatt3R Gaussian predictor."""

    def __init__(self, config):
        """Init.

        Args:
            config: The config.
        """
        super().__init__()
        self.config = config

        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="gaussian_head",
            output_mode="pts3d+gaussian+desc24",
            depth_mode=("exp", -mast3r_model.inf, mast3r_model.inf),
            conf_mode=("exp", 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, "sh_degree") else 1,
        )
        self.encoder.requires_grad_(False)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
        from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat import (
            decoder_splatting_cuda as pixelsplat_decoder,
        )

        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])

    def forward(self, view1, view2):
        """Forward.

        Args:
            view1: The view1.
            view2: The view2.
        """
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1["covariances"] = geometry.build_covariance(pred1["scales"], pred1["rotations"])
        pred2["covariances"] = geometry.build_covariance(pred2["scales"], pred2["rotations"])

        new_sh1 = torch.zeros_like(pred1["sh"])
        new_sh2 = torch.zeros_like(pred2["sh"])
        new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1["original_img"], "b c h w -> b h w c"))
        new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2["original_img"], "b c h w -> b h w c"))
        pred1["sh"] = pred1["sh"] + new_sh1
        pred2["sh"] = pred2["sh"] + new_sh2

        pred2["pts3d_in_other_view"] = pred2.pop("pts3d")
        pred2["means_in_other_view"] = pred2.pop("means")
        return pred1, pred2


__all__ = ["MAST3RGaussians"]
