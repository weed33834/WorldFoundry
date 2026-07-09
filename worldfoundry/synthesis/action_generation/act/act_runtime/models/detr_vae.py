# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

from typing import Any

import numpy as np

from .backbone import build_backbone
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


def reparametrize(mu: Any, logvar: Any):
    """Sample the ACT latent variable with the VAE reparameterization trick.

    Args:
        mu: Latent mean tensor.
        logvar: Latent log-variance tensor.
    """
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int):
    """Create the fixed sinusoidal table used by the ACT encoder.

    Args:
        n_position: Number of sequence positions.
        d_hid: Hidden size for each position.
    """
    import torch

    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def build_encoder(args: Any):
    """Build the ACT latent encoder transformer.

    Args:
        args: Namespace-like object with encoder hyperparameters.
    """
    from torch import nn

    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)


def build(args: Any):
    """Build the ACT DETR-VAE policy network.

    Args:
        args: Namespace-like object with ACT policy hyperparameters.
    """
    model = _build_detr_vae_class()(
        [build_backbone(args)],
        build_transformer(args),
        build_encoder(args),
        state_dim=int(getattr(args, "state_dim", 14)),
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
    )
    return model


def _build_detr_vae_class():
    import torch
    from torch import nn

    class DETRVAE(nn.Module):
        def __init__(self, backbones, transformer, encoder, state_dim: int, num_queries: int, camera_names: list[str]) -> None:
            super().__init__()
            self.num_queries = num_queries
            self.camera_names = camera_names
            self.transformer = transformer
            self.encoder = encoder
            hidden_dim = transformer.d_model
            self.action_head = nn.Linear(hidden_dim, state_dim)
            self.is_pad_head = nn.Linear(hidden_dim, 1)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.latent_dim = 32
            self.cls_embed = nn.Embedding(1, hidden_dim)
            self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
            self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
            self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
            is_training = actions is not None
            bs, _ = qpos.shape
            if is_training:
                action_embed = self.encoder_action_proj(actions)
                qpos_embed = torch.unsqueeze(self.encoder_joint_proj(qpos), axis=1)
                cls_embed = torch.unsqueeze(self.cls_embed.weight, axis=0).repeat(bs, 1, 1)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1).permute(1, 0, 2)
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
                pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)[0]
                latent_info = self.latent_proj(encoder_output)
                mu = latent_info[:, : self.latent_dim]
                logvar = latent_info[:, self.latent_dim :]
                latent_input = self.latent_out_proj(reparametrize(mu, logvar))
            else:
                mu = logvar = None
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)

            all_cam_features = []
            all_cam_pos = []
            for cam_id, _cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                all_cam_features.append(self.input_proj(features[0]))
                all_cam_pos.append(pos[0])
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
            return self.action_head(hs), self.is_pad_head(hs), [mu, logvar]

    return DETRVAE
