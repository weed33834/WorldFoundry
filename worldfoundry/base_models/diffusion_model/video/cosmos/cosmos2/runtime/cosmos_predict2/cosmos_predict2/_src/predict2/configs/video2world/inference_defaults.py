# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight defaults for Predict2 inference configuration composition."""

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration import CheckpointConfig, ObjectStoreConfig
from worldfoundry.core.configuration.lazy_config import LazyCall as L


def create_rectified_flow_model(config: dict):
    """Import the heavy model implementation only when inference instantiates it."""

    from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
        Video2WorldModelRectifiedFlow,
        Video2WorldModelRectifiedFlowConfig,
    )

    return Video2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlowConfig(**config))


_BASE_NET = L("cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit.MinimalV1LVGDiT")(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=L("cosmos_predict2._src.predict2.networks.minimal_v4_dit.SACConfig")(),
)
_NET_14B = deepcopy(_BASE_NET)
_NET_14B.model_channels = 5120
_NET_14B.num_heads = 40
_NET_14B.num_blocks = 36

_CONDITIONER = L("cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner.Video2WorldConditioner")(
    fps=L("cosmos_predict2._src.predict2.conditioner.ReMapkey")(
        input_key="fps", output_key="fps", dropout_rate=0.0, dtype=None
    ),
    padding_mask=L("cosmos_predict2._src.predict2.conditioner.ReMapkey")(
        input_key="padding_mask", output_key="padding_mask", dropout_rate=0.0, dtype=None
    ),
    text=L("cosmos_predict2._src.predict2.conditioner.TextAttr")(
        input_key=["t5_text_embeddings"], dropout_rate=0.2, use_empty_string=False
    ),
    use_video_condition=L("cosmos_predict2._src.predict2.conditioner.BooleanFlag")(
        input_key="fps", output_key="use_video_condition", dropout_rate=0.2
    ),
)


def register_inference_defaults() -> None:
    store = ConfigStore.instance()
    store.store(
        group="model",
        package="_global_",
        name="fsdp_rectified_flow",
        node={
            "model": L(create_rectified_flow_model)(
                config={"fsdp_shard_size": 8, "state_t": 24},
                _recursive_=False,
            )
        },
    )
    store.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=_BASE_NET)
    store.store(group="net", package="model.config.net", name="cosmos_v1_14B", node=_NET_14B)
    store.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=_CONDITIONER,
    )
    store.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="wan2pt1_tokenizer",
        node=L("cosmos_predict2._src.predict2.tokenizers.wan2pt1.Wan2pt1VAEInterface")(name="wan2pt1_tokenizer"),
    )
    store.store(
        group="checkpoint",
        package="checkpoint",
        name="s3",
        node=CheckpointConfig(
            load_from_object_store=ObjectStoreConfig(
                enabled=False,
                credentials="credentials/s3_checkpoint.secret",
                bucket="bucket",
            )
        ),
    )


__all__ = ["create_rectified_flow_model", "register_inference_defaults"]
