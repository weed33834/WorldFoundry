"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention


def exists(val):
    """Exists.

    Args:
        val: The val.
    """
    return val is not None


def default(val, d):
    """Default.

    Args:
        val: The val.
        d: The d.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


class SwiGLU(nn.Module):
    """Swi glu implementation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class GEGLU(nn.Module):
    """Geglu implementation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class MLP(nn.Module):
    """Mlp implementation."""
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ):
        """Init.

        Args:
            input_dim: The input dim.
            expansion: The expansion.
            dropout: The dropout.
            gated: The gated.
            output_dim: The output dim.
        """
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x


class LayerScale(nn.Module):
    """Layer scale implementation."""
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
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


class AttentionBlock(nn.Module):
    """Attention block implementation."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            expansion: The expansion.
            dropout: The dropout.
            cosine: The cosine.
            gated: The gated.
            layer_scale: The layer scale.
            context_dim: The context dim.
            use_bias: The use bias.
        """
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=use_bias)
        self.q = nn.Linear(dim, dim, bias=use_bias)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=use_bias)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attn.

        Args:
            x: The x.
            attn_bias: The attn bias.
            context: The context.
            pos_embed: The pos embed.
            pos_embed_context: The pos embed context.

        Returns:
            The return value.
        """
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if pos_embed is not None:
            pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
            q = q + pos_embed
        if pos_embed_context is not None:
            pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads)
            k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = _worldfoundry_scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.
            attn_bias: The attn bias.
            context: The context.
            pos_embed: The pos embed.
            pos_embed_context: The pos embed context.

        Returns:
            The return value.
        """
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class AttentionLayer(nn.Module):
    """Attention layer implementation."""
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        """Init.

        Args:
            num_blocks: The num blocks.
            dim: The dim.
            num_heads: The num heads.
            expansion: The expansion.
            dropout: The dropout.
            cosine: The cosine.
            gated: The gated.
            layer_scale: The layer scale.
            context_dim: The context dim.
            use_bias: The use bias.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    cosine=cosine,
                    gated=gated,
                    layer_scale=layer_scale,
                    context_dim=context_dim,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.
            context: The context.
            pos_embed: The pos embed.
            pos_embed_context: The pos embed context.
            attn_bias: The attn bias.

        Returns:
            The return value.
        """
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                attn_bias=attn_bias,
            )
        return x


class PositionEmbeddingSine(nn.Module):
    """Position embedding sine implementation."""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """Init.

        Args:
            num_pos_feats: The num pos feats.
            temperature: The temperature.
            normalize: The normalize.
            scale: The scale.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.
            mask: The mask.

        Returns:
            The return value.
        """
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        """Repr.

        Args:
            _repr_indent: The repr indent.
        """
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class ResidualConvUnit(nn.Module):
    """Residual conv unit implementation."""
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        dilation: int = 1,
        layer_scale: float = 1.0,
        use_norm: bool = False,
    ):
        """Init.

        Args:
            dim: The dim.
            kernel_size: The kernel size.
            padding_mode: The padding mode.
            dilation: The dilation.
            layer_scale: The layer scale.
            use_norm: The use norm.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.activation = nn.LeakyReLU()
        self.gamma = nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1)) if layer_scale > 0.0 else 1.0
        self.norm1 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.gamma * out + x


class ResUpsampleBil(nn.Module):
    """Res upsample bil implementation."""
    def __init__(
        self,
        hidden_dim,
        output_dim: int = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        use_norm: bool = False,
        **kwargs,
    ):
        """Init.

        Args:
            hidden_dim: The hidden dim.
            output_dim: The output dim.
            num_layers: The num layers.
            kernel_size: The kernel size.
            layer_scale: The layer scale.
            padding_mode: The padding mode.
            use_norm: The use norm.
        """
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim // 2
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x: The x.
        """
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x
