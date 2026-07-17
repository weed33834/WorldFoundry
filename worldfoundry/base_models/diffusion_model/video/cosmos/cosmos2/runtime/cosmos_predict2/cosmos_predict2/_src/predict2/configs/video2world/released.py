# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only configuration entries for released Predict2 models."""

from copy import deepcopy

from hydra.core.config_store import ConfigStore

TEXT_ENCODER_CONFIG = {
    "embedding_concat_strategy": "full_concat",
    "compute_online": True,
    "ckpt_path": "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
    "s3_credential_path": "credentials/s3_checkpoint.secret",
}


def released_config(*, size: str, name: str) -> dict:
    is_14b = size == "14B"
    return {
        "defaults": [
            {"override /model": "fsdp_rectified_flow"},
            {"override /net": f"cosmos_v1_{size}"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /checkpoint": "s3"},
            "_self_",
        ],
        "job": {"project": "cosmos_diffusion_v2", "group": "inference", "name": name},
        "model": {
            "config": {
                "min_num_conditional_frames": 0,
                "max_num_conditional_frames": 2,
                "conditional_frames_probs": {0: 0.5, 1: 0.25, 2: 0.25},
                "fsdp_shard_size": 32 if is_14b else 8,
                "resolution": "720",
                "state_t": 24,
                "shift": 5,
                "use_dynamic_shift": False,
                "net": {
                    "rope_enable_fps_modulation": False,
                    "rope_h_extrapolation_ratio": 3.0,
                    "rope_w_extrapolation_ratio": 3.0,
                    "rope_t_extrapolation_ratio": 1.0,
                    "timestep_scale": 0.001,
                    "sac_config": {"mode": f"predict2_{size.lower()}_720_aggressive"},
                    "use_crossattn_projection": True,
                    "crossattn_proj_in_channels": 100352,
                    "crossattn_emb_channels": 1024,
                    "use_wan_fp32_strategy": True,
                },
                "conditioner": {
                    "use_video_condition": {"dropout_rate": 0.0},
                    "text": {"dropout_rate": 0.2, "use_empty_string": False},
                },
                "tokenizer": {"temporal_window": 16},
                "text_encoder_class": "reason1p1_7B",
                "text_encoder_config": deepcopy(TEXT_ENCODER_CONFIG),
            }
        },
        "model_parallel": {"context_parallel_size": 8 if is_14b else 2},
    }


PRETRAINED_2B = "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2"
POSTTRAINED_2B = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
RELEASED_14B = "Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma"


def register_released_experiments() -> None:
    store = ConfigStore.instance()
    pretrained_2b = released_config(size="2B", name=PRETRAINED_2B)
    store.store(group="experiment", package="_global_", name=PRETRAINED_2B, node=pretrained_2b)

    posttrained_2b = released_config(size="2B", name=POSTTRAINED_2B)
    posttrained_2b["model"]["config"].update(
        conditional_frame_timestep=0.1,
        use_kerras_sigma_at_inference=True,
    )
    store.store(group="experiment", package="_global_", name=POSTTRAINED_2B, node=posttrained_2b)

    released_14b = released_config(size="14B", name=RELEASED_14B)
    released_14b["model"]["config"]["use_high_sigma_strategy"] = True
    store.store(group="experiment", package="_global_", name=RELEASED_14B, node=released_14b)


__all__ = ["TEXT_ENCODER_CONFIG", "register_released_experiments", "released_config"]
