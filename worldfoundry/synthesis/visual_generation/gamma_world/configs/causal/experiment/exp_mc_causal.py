"""Released Gamma causal inference configuration."""

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyDict
from worldfoundry.synthesis.visual_generation.gamma_world.configs.causal.experiment.exp_mc import (
    _NET,
    _model_config,
)

net = deepcopy(_NET)
net.update({"use_sparse_hub": True, "z_num": 8})
model = _model_config(net)
model.update(
    {
        "noise_scheme": "diffusion_forcing",
        "num_frame_per_block": 3,
        "max_latent_frames_per_gpu": 24,
        "split_cp_in_model": False,
    }
)

CAUSAL = LazyDict(
    {
        "defaults": [
            {"override /model": "fsdp_mv"},
            {"override /net": "cosmos_v2_2b_causal"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        "job": {"group": "causal_cosmos2", "name": "causal"},
        "model": {"config": model},
        "model_parallel": {"context_parallel_size": 1},
    }
)

ConfigStore.instance().store(group="experiment", package="_global_", name=CAUSAL["job"]["name"], node=CAUSAL)
