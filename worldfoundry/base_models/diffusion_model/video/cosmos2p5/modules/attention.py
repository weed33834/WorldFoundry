"""Module for base_models -> diffusion_model -> video -> cosmos2p5 -> modules -> attention.py functionality."""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from worldfoundry.core.distributed.sequence_parallel_runtime import all_to_all, get_sequence_parallel_group, split_forward_gather_backward
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention


class Attention(nn.Module):
    """Attention implementation."""
    def __init__(self, backend='sdpa', qkv_format='bhsd', is_selfattn=True):
        """Init.

        Args:
            backend: The backend.
            qkv_format: The qkv format.
            is_selfattn: The is selfattn.
        """
        super().__init__()
        assert backend in ['sdpa', 'sage']
        assert qkv_format in ['bshd', 'bhsd', 'sbhd']
        self.backend = backend
        self.qkv_format = qkv_format
        self.is_selfattn = is_selfattn

    def forward(self, query, key, value, **kwargs):
        """Forward.

        Args:
            query: The query.
            key: The key.
            value: The value.
        """
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            seq_dim = self.qkv_format.index('s')
            head_dim = self.qkv_format.index('h')
            query = all_to_all(query, scatter_dim=head_dim, gather_dim=seq_dim, group=sp_group)
            if self.is_selfattn:
                key = all_to_all(key, scatter_dim=head_dim, gather_dim=seq_dim, group=sp_group)
                value = all_to_all(value, scatter_dim=head_dim, gather_dim=seq_dim, group=sp_group)
            else:
                key = split_forward_gather_backward(key, dim=head_dim, group=sp_group)
                value = split_forward_gather_backward(value, dim=head_dim, group=sp_group)
        if self.backend in ['sdpa', 'sage']:
            if self.qkv_format != 'bhsd':
                src_format = ' '.join([char for char in self.qkv_format])
                query = rearrange(query, f'{src_format} -> b h s d')
                key = rearrange(key, f'{src_format} -> b h s d')
                value = rearrange(value, f'{src_format} -> b h s d')
            if self.backend == 'sdpa':
                out = _worldfoundry_scaled_dot_product_attention(query, key, value, **kwargs)
            else:
                from sageattention import sageattn

                out = sageattn(query, key, value, **kwargs)
            if self.qkv_format != 'bhsd':
                out = rearrange(out, f'b h s d -> {src_format}')
        else:
            assert False
        if sp_group is not None:
            out = all_to_all(out, scatter_dim=seq_dim, gather_dim=head_dim, group=sp_group)
        return out
