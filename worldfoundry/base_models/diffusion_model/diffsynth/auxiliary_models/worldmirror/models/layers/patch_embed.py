# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

"""Module for base_models -> diffusion_model -> diffsynth -> auxiliary_models -> worldmirror -> models -> layers -> patch_embed.py functionality."""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

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
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
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


class PatchEmbed_Mlp(PatchEmbed):
    """Patch embed mlp implementation."""
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 norm_layer=None, 
                 flatten_embedding=True):
        """Init.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            in_chans: The in chans.
            embed_dim: The embed dim.
            norm_layer: The norm layer.
            flatten_embedding: The flatten embedding.
        """
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten_embedding)

        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0,2,3,1)),
            Mlp(in_chans * patch_size**2, 4*embed_dim, embed_dim),
            Permute((0,3,1,2)),
            )
    

class PixelUnshuffle (nn.Module):
    """Pixel unshuffle implementation."""
    def __init__(self, downscale_factor):
        """Init.

        Args:
            downscale_factor: The downscale factor.
        """
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """Forward.

        Args:
            input: The input.
        """
        if input.numel() == 0:
            # this is not in the original torch implementation
            C,H,W = input.shape[-3:]
            assert H and W and H % self.downscale_factor == W%self.downscale_factor == 0
            return input.view(*input.shape[:-3], C*self.downscale_factor**2, H//self.downscale_factor, W//self.downscale_factor)
        else:
            return F.pixel_unshuffle(input, self.downscale_factor)
        

class Permute(torch.nn.Module):
    """Permute implementation."""
    dims: tuple[int, ...]
    def __init__(self, dims: tuple[int, ...]) -> None:
        """Init.

        Args:
            dims: The dims.

        Returns:
            The return value.
        """
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        """Repr."""
        return f"Permute{self.dims}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            input: The input.

        Returns:
            The return value.
        """
        return input.permute(*self.dims)


def _ntuple(n):
    """Helper function to ntuple.

    Args:
        n: The n.
    """
    def parse(x):
        """Parse.

        Args:
            x: The x.
        """
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        """Init.

        Args:
            in_features: The in features.
            hidden_features: The hidden features.
            out_features: The out features.
            act_layer: The act layer.
            bias: The bias.
            drop: The drop.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

