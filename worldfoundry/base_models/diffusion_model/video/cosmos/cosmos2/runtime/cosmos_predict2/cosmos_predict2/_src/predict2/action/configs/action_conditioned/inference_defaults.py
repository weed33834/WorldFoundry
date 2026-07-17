# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight action-conditioned inference configuration."""

from copy import deepcopy

from cosmos_predict2._src.predict2.configs.video2world.inference_defaults import register_inference_defaults
from cosmos_predict2._src.predict2.configs.video2world.released import released_config
from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyCall as L

BRIDGE_EXPERIMENT = "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"
GR00T_EXPERIMENT = "cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release"
DREAMDOJO_EXPERIMENT = "dreamdojo_2b_480_640_gr1"


def create_action_model(config: dict):
    from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
        ActionVideo2WorldModelRectifiedFlow,
        Video2WorldModelRectifiedFlowConfig,
    )

    return ActionVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlowConfig(**config))


def _action_net(*, chunked: bool) -> dict:
    target = "cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit." + (
        "ActionChunkConditionedMinimalV1LVGDiT" if chunked else "ActionConditionedMinimalV1LVGDiT"
    )
    return L(target)(
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


def _conditioner() -> dict:
    common = {
        "fps": L("cosmos_predict2._src.predict2.conditioner.ReMapkey")(
            input_key="fps", output_key="fps", dropout_rate=0.0, dtype=None
        ),
        "padding_mask": L("cosmos_predict2._src.predict2.conditioner.ReMapkey")(
            input_key="padding_mask", output_key="padding_mask", dropout_rate=0.0, dtype=None
        ),
        "text": L("cosmos_predict2._src.predict2.conditioner.TextAttr")(
            input_key=["t5_text_embeddings"], dropout_rate=0.2, use_empty_string=False
        ),
        "use_video_condition": L("cosmos_predict2._src.predict2.conditioner.BooleanFlag")(
            input_key="fps", output_key="use_video_condition", dropout_rate=0.2
        ),
        "action": L("cosmos_predict2._src.predict2.conditioner.ReMapkey")(
            input_key="action", output_key="action", dropout_rate=0.0, dtype=None
        ),
    }
    return L(
        "cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner.ActionConditionedConditioner"
    )(**common)


def _experiment(name: str, *, action_dim: int) -> dict:
    config = released_config(size="2B", name=name)
    config["defaults"] = [
        {"override /model": "action_rectified_flow"},
        {"override /net": "action_chunk_2B"},
        {"override /conditioner": "action_conditioner"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /checkpoint": "s3"},
        "_self_",
    ]
    config["job"].update(project="cosmos_predict2_action_conditioned", group="inference")
    config["model"]["config"].update(
        min_num_conditional_frames=1,
        max_num_conditional_frames=1,
        conditional_frames_probs=None,
        state_t=4,
    )
    config["model"]["config"]["net"].update(
        action_dim=action_dim,
        temporal_compression_ratio=4,
        num_action_per_chunk=12,
        zero_init_action_embedder=False,
    )
    config["model_parallel"] = {"context_parallel_size": 1}
    return config


def register_action_inference() -> None:
    register_inference_defaults()
    store = ConfigStore.instance()
    store.store(
        group="model",
        package="_global_",
        name="action_rectified_flow",
        node={
            "model": L(create_action_model)(
                config={"fsdp_shard_size": 8, "state_t": 4},
                _recursive_=False,
            )
        },
    )
    store.store(group="net", package="model.config.net", name="action_2B", node=_action_net(chunked=False))
    store.store(group="net", package="model.config.net", name="action_chunk_2B", node=_action_net(chunked=True))
    store.store(
        group="conditioner",
        package="model.config.conditioner",
        name="action_conditioner",
        node=_conditioner(),
    )
    for name, action_dim in ((BRIDGE_EXPERIMENT, 7), (GR00T_EXPERIMENT, 384), (DREAMDOJO_EXPERIMENT, 29)):
        store.store(
            group="experiment",
            package="_global_",
            name=name,
            node=deepcopy(_experiment(name, action_dim=action_dim)),
        )


__all__ = [
    "BRIDGE_EXPERIMENT",
    "DREAMDOJO_EXPERIMENT",
    "GR00T_EXPERIMENT",
    "register_action_inference",
]
