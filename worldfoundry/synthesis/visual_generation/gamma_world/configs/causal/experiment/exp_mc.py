"""Released Gamma bidirectional inference configuration."""

from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyDict


def _model_config(net: dict) -> dict:
    return {
        "noise_scheme": "consistent_noise",
        "num_frame_per_block": 48,
        "max_latent_frames_per_gpu": 48,
        "denoise_replace_gt_frames": True,
        "state_t": 48,
        "fsdp_shard_size": 8,
        "shift": 5,
        "split_cp_in_model": True,
        "net": net,
        "conditioner": {
            "use_video_condition": {"dropout_rate": 0.0},
            "text": {"dropout_rate": 0.0, "use_empty_string": False},
        },
        "text_encoder_class": "reason1p1_7B",
        "text_encoder_config": {
            "embedding_concat_strategy": "full_concat",
            "compute_online": True,
            "ckpt_path": "hf://nvidia/Cosmos-Reason1-7B",
        },
        "use_action_control": True,
        "condition_locations": ["first_random_n"],
        "min_num_conditional_frames_per_view": 1,
        "max_num_conditional_frames_per_view": 1,
        "min_num_conditional_frames": 1,
        "max_num_conditional_frames": 1,
        "shuffle_agents": True,
    }


_NET = {
    "rope_enable_fps_modulation": False,
    "rope_h_extrapolation_ratio": 2.0,
    "rope_w_extrapolation_ratio": 2.0,
    "rope_t_extrapolation_ratio": 1.0,
    "timestep_scale": 0.001,
    "enable_action_control": True,
    "action_keyboard_dim": 23,
    "action_camera_dim": 2,
    "action_use_camera": True,
    "action_embed_dim": 256,
    "action_temporal_downsample": 4,
    "use_multi_agent_rope": True,
    "multi_agent_rope_num_agents": 2,
    "multi_agent_rope_agent_id_offset": 0,
    "multi_agent_rope_simplex_pool_size": 4,
    "multi_agent_rope_agent_encoding": "simplex",
    "multi_agent_rope_agent_scale": 1.0,
    "multi_agent_rope_share_action_encoder": True,
}

BIDIRECTIONAL = LazyDict(
    {
        "defaults": [
            {"override /model": "fsdp_mv"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        "job": {"group": "causal_cosmos2", "name": "bidirectional"},
        "model": {"config": _model_config(_NET)},
        "model_parallel": {"context_parallel_size": 1},
    }
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name=BIDIRECTIONAL["job"]["name"], node=BIDIRECTIONAL
)
