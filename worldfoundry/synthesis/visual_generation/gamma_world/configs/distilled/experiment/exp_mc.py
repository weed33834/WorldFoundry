"""Released Gamma causal-few-step inference configuration."""

from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyDict

CAUSAL_FEW_STEP = LazyDict(
    {
        "defaults": [
            {"override /net": "causal_cosmosv2_2b"},
            {"override /model": "fsdp_mv"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            "_self_",
        ],
        "job": {"group": "causal_cosmos2", "name": "causal_few_step"},
        "model": {
            "config": {
                "state_t": 48,
                "min_num_conditional_frames": 1,
                "max_num_conditional_frames": 1,
                "denoise_replace_gt_frames": True,
                "context_noise": 128,
                "use_action_control": True,
                "model_type": "i2v",
                "i2v_zero_latent_condition": True,
                "fsdp_shard_size": 64,
                "shuffle_agents": True,
                "conditioner": {
                    "use_video_condition": {"dropout_rate": 0.0},
                    "text": {"dropout_rate": 0.0, "use_empty_string": False},
                },
                "net": {
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
                    "use_sparse_hub": True,
                    "z_num": 8,
                    "multi_agent_rope_simplex_pool_size": 4,
                    "multi_agent_rope_agent_encoding": "simplex",
                    "multi_agent_rope_agent_scale": 1.0,
                    "multi_agent_rope_share_action_encoder": True,
                    "local_attn_size": 24,
                    "sink_size": 0,
                },
            }
        },
        "model_parallel": {"context_parallel_size": 1},
    },
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=CAUSAL_FEW_STEP["job"]["name"],
    node=CAUSAL_FEW_STEP,
)
