"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import contextlib
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.xformers import index_select_cat, memory_efficient_attention, scaled_index_add
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

fmha = None
XFORMERS_AVAILABLE = False


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    """Named apply.

    Args:
        fn: The fn.
        module: The module.
        name: The name.
        depth_first: The depth first.
        include_root: The include root.

    Returns:
        The return value.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def get_parameter_groups(model, lr, wd=1e-5, ld=0.9, skip_list=()):
    """Get parameter groups.

    Args:
        model: The model.
        lr: The lr.
        wd: The wd.
        ld: The ld.
        skip_list: The skip list.
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    skip = {}
    if skip_list is not None:
        skip = skip_list
    elif hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    num_layers = model.n_blocks
    layer_scale = list(ld ** (num_layers - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1:  # norm
            group_name = "no_decay"
            this_wd = 0.0
        # layer scale, bias beta?
        elif name in skip or name.endswith(".gamma") or name.endswith(".beta") or name.endswith(".bias"):
            group_name = "no_decay"
            this_wd = 0.0
        elif "cls_token" in name or "pos_embed" in name or "mask_token" in name:
            group_name = "no_decay"
            this_wd = 0.0
        else:
            group_name = "decay"
            this_wd = wd

        if name.startswith("blocks"):
            layer_id = int(name.split(".")[1])
        elif name.startswith("patch_embed"):
            layer_id = 0
        else:
            layer_id = 0

        group_name = f"layer_{layer_id}_{group_name}"

        if group_name not in parameter_group_names:
            scale = layer_scale[layer_id]
            cur_lr = lr * scale

            parameter_group_names[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values()), [v["lr"] for k, v in parameter_group_vars.items()]


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_2tuple(x):
    """Make 2tuple.

    Args:
        x: The x.
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    """Drop add residual stochastic depth.

    Args:
        x: The x.
        residual_func: The residual func.
        sample_drop_ratio: The sample drop ratio.

    Returns:
        The return value.
    """
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    """Get branges scales.

    Args:
        x: The x.
        sample_drop_ratio: The sample drop ratio.
    """
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    if fmha is None:
        raise NotImplementedError("Nested tensor attention requires xFormers, which is disabled in ViPE")

    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    """Add residual.

    Args:
        x: The x.
        brange: The brange.
        residual: The residual.
        residual_scale_factor: The residual scale factor.
        scaling_vector: The scaling vector.
    """
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


def drop_add_residual_stochastic_depth_list(
    x_list: List[torch.Tensor],
    residual_func: Callable[[torch.Tensor, Any], torch.Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> torch.Tensor:
    """Drop add residual stochastic depth list.

    Args:
        x_list: The x list.
        residual_func: The residual func.
        sample_drop_ratio: The sample drop ratio.
        scaling_vector: The scaling vector.

    Returns:
        The return value.
    """
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """Init.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            in_chans: The in chans.
            embed_dim: The embed dim.
            norm_layer: The norm layer.
            flatten_embedding: The flatten embedding.

        Returns:
            The return value.
        """
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Flops.

        Returns:
            The return value.
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop path.

    Args:
        x: The x.
        drop_prob: The drop prob.
        training: The training.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        """Init.

        Args:
            drop_prob: The drop prob.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    """Layer scale implementation."""
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Init.

        Args:
            dim: The dim.
            init_values: The init values.
            inplace: The inplace.

        Returns:
            The return value.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    """Attention implementation."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            qkv_bias: The qkv bias.
            proj_bias: The proj bias.
            attn_drop: The attn drop.
            proj_drop: The proj drop.

        Returns:
            The return value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = _worldfoundry_scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """Mem eff attention implementation."""
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.
            attn_bias: The attn bias.

        Returns:
            The return value.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = torch.unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Mlp implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            drop: The drop.
            bias: The bias.

        Returns:
            The return value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFNFused(nn.Module):
    """Swi gluffn fused implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            drop: The drop.
            bias: The bias.

        Returns:
            The return value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Block(nn.Module):
    """Block implementation."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            mlp_ratio: The mlp ratio.
            qkv_bias: The qkv bias.
            proj_bias: The proj bias.
            ffn_bias: The ffn bias.
            drop: The drop.
            attn_drop: The attn drop.
            init_values: The init values.
            drop_path: The drop path.
            act_layer: The act layer.
            norm_layer: The norm layer.
            attn_class: The attn class.
            ffn_layer: The ffn layer.

        Returns:
            The return value.
        """
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            """Attn residual func.

            Args:
                x: The x.

            Returns:
                The return value.
            """
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            """Ffn residual func.

            Args:
                x: The x.

            Returns:
                The return value.
            """
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(Block):
    """Nested tensor block implementation."""
    def forward_nested(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                """Attn residual func.

                Args:
                    x: The x.
                    attn_bias: The attn bias.

                Returns:
                    The return value.
                """
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                """Ffn residual func.

                Args:
                    x: The x.
                    attn_bias: The attn bias.

                Returns:
                    The return value.
                """
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                """Attn residual func.

                Args:
                    x: The x.
                    attn_bias: The attn bias.

                Returns:
                    The return value.
                """
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                """Ffn residual func.

                Args:
                    x: The x.
                    attn_bias: The attn bias.

                Returns:
                    The return value.
                """
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        """Forward.

        Args:
            x_or_x_list: The x or x list.
        """
        if isinstance(x_or_x_list, torch.Tensor):
            return super(NestedTensorBlock, self).forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            raise NotImplementedError("Nested tensor attention requires xFormers, which is disabled in ViPE")
        else:
            raise AssertionError


class BlockChunk(nn.ModuleList):
    """Block chunk implementation."""
    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        for block in self:
            x = block(x)
        return x


class DinoVisionTransformer(nn.Module):
    """Dino vision transformer implementation."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        output_idx=[5, 12, 18, 24],
        checkpoint: bool = False,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.0,
        use_norm=False,
        frozen_stages=0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.embed_dims = [embed_dim] * output_idx[-1]
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.depths = output_idx
        self.checkpoint = checkpoint
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, max(1, num_register_tokens), embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                """F."""
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = nn.LayerNorm(embed_dim)
        self.use_norm = use_norm
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.num_register_tokens:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate pos encoding.

        Args:
            x: The x.
            w: The w.
            h: The h.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """Prepare tokens with masks.

        Args:
            x: The x.
            masks: The masks.
        """
        B, nc, w, h = x.shape
        with torch.no_grad() if self.frozen_stages > -1 else contextlib.nullcontext():
            x = self.patch_embed(x)
        if masks is not None:
            masks = masks.bool().view(B, -1, 1)
            x = torch.where(masks, self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.num_register_tokens:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward(self, x, masks=None):
        """Forward.

        Args:
            x: The x.
            masks: The masks.
        """
        shapes = [val // self.patch_size for val in x.shape[-2:]]
        batch_size = x.shape[0]
        x = self.prepare_tokens_with_masks(x, masks)
        outputs = []
        for i, blk in enumerate(self.blocks):
            with torch.no_grad() if i < self.frozen_stages else contextlib.nullcontext():
                x = blk(x)
            outputs.append(x)

        if self.use_norm:
            with torch.no_grad() if self.frozen_stages >= len(self.blocks) else contextlib.nullcontext():
                outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, :1] for out in outputs]
        outputs = [out[:, self.num_register_tokens + 1 :] for out in outputs]
        outputs = [out.reshape(batch_size, *shapes, -1) for out in outputs]

        return (outputs, class_tokens)

    def get_params(self, lr, wd, ld, *args, **kwargs):
        """Get params.

        Args:
            lr: The lr.
            wd: The wd.
            ld: The ld.
        """
        encoder_p, encoder_lr = get_parameter_groups(self, lr, wd, ld)
        return encoder_p, encoder_lr

    def freeze(self) -> None:
        """Freeze.

        Returns:
            The return value.
        """
        for module in self.modules():
            module.eval()
        for parameters in self.parameters():
            parameters.requires_grad = False

    def train(self, mode=True):
        """Train.

        Args:
            mode: The mode.
        """
        super().train(mode)
        if self.frozen_stages > -1:
            for p in self.patch_embed.parameters():
                p.requires_grad = False

        for i, blk in enumerate(self.blocks):
            if i < self.frozen_stages:
                blk.eval()
                for p in blk.parameters():
                    p.requires_grad = False

        for p in self.norm.parameters():
            p.requires_grad = self.frozen_stages <= len(self.blocks) and self.use_norm

        self.cls_token.requires_grad = self.frozen_stages < 1
        self.pos_embed.requires_grad = self.frozen_stages < 1
        self.mask_token.requires_grad = False
        self.register_tokens.requires_grad = False


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    """Helper function to make dinov2 model name.

    Args:
        arch_name: The arch name.
        patch_size: The patch size.

    Returns:
        The return value.
    """
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def vit_small(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Vit small.

    Args:
        patch_size: The patch size.
        num_register_tokens: The num register tokens.
        export: The export.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(NestedTensorBlock, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Vit base.

    Args:
        patch_size: The patch size.
        num_register_tokens: The num register tokens.
        export: The export.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(NestedTensorBlock, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Vit large.

    Args:
        patch_size: The patch size.
        num_register_tokens: The num register tokens.
        export: The export.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(NestedTensorBlock, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: str = "",
    output_idx: Sequence[int] = [],
    num_register_tokens: int = 0,
    drop_path_rate: float = 0.0,
    use_norm: bool = False,
    export: bool = False,
    interpolate_offset: float = 0.0,
    frozen_stages: int = 0,
    **kwargs,
):
    """Helper function to make dinov2 model."""
    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        output_idx=output_idx,
        drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        use_norm=use_norm,
        export=export,
        interpolate_offset=interpolate_offset,
        frozen_stages=frozen_stages,
    )
    vit_kwargs.update(**kwargs)
    model = eval(arch_name)(**vit_kwargs)
    if pretrained == "":
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}"
        if num_register_tokens > 0:
            url += "_reg4"
        url += "_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        model.load_state_dict(state_dict, strict=False)
        # print(info)
    elif pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        # print(f"loading from {pretrained} with:", info)
    # else:
    # print("Not loading pretrained weights for backbone")
    return model


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit
