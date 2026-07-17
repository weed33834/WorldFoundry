"""Inference-only Gamma model construction and safetensors loading."""

from __future__ import annotations

import importlib

import torch

from worldfoundry.core.configuration.hydra import get_config_module, override
from worldfoundry.core.configuration.lazy_config import instantiate
from worldfoundry.core.distributed import torch_process_group as distributed
from worldfoundry.core.distributed.logging import log
from worldfoundry.core.io.easy_io import resolve_checkpoint_path
from worldfoundry.core.utils import inference_runtime as misc


def _load_safetensors(net: torch.nn.Module, checkpoint_path: str) -> None:
    from safetensors.torch import load_file

    loaded = load_file(resolve_checkpoint_path(checkpoint_path), device="cpu")
    target = net.state_dict()
    state = {}
    unexpected = []
    mismatched = []
    for key, value in loaded.items():
        target_key = key
        for prefix in ("model.net.", "net."):
            if target_key.startswith(prefix):
                target_key = target_key[len(prefix) :]
                break
        if (
            target_key.endswith("_extra_state")
            or target_key.endswith("pos_embedder.seq")
            or target_key.startswith("accum_")
        ):
            continue
        if target_key not in target:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target[target_key].shape):
            mismatched.append(f"{key}: {tuple(value.shape)} vs {tuple(target[target_key].shape)}")
        else:
            state[target_key] = value
    missing = [
        key
        for key in target
        if key not in state
        and not key.endswith("_extra_state")
        and not key.endswith("pos_embedder.seq")
        and not key.startswith("accum_")
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Gamma safetensors mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, mismatched={mismatched[:10]}"
        )
    net.load_state_dict(state, strict=False)


def load_model_from_checkpoint(
    experiment_name,
    s3_checkpoint_dir,
    config_file,
    enable_fsdp=False,
    seed=0,
    experiment_opts=None,
    vae_pth=None,
    text_encoder_pth=None,
    **kwargs,
):
    """Instantiate one released Gamma network and load its inference weights."""

    del kwargs
    if not str(s3_checkpoint_dir).endswith(".safetensors"):
        raise ValueError("the inference-only Gamma runtime accepts released .safetensors networks only")
    module = importlib.import_module(get_config_module(config_file))
    config = module.make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + list(experiment_opts or []))
    config.checkpoint.load_path = str(s3_checkpoint_dir)
    if vae_pth is not None:
        config.model.config.tokenizer.vae_pth = vae_pth
    if text_encoder_pth is not None:
        config.model.config.text_encoder_config.ckpt_path = text_encoder_pth
    if not enable_fsdp:
        config.model.config.fsdp_shard_size = 1
    config.freeze()
    misc.set_random_seed(seed=seed, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    model = instantiate(config.model)
    model.prepare_inference()
    if distributed.is_rank0():
        log.info("loading Gamma safetensors from {}", s3_checkpoint_dir)
        _load_safetensors(model.net, str(s3_checkpoint_dir))
    distributed.sync_model_states(model, src=0)
    torch.cuda.empty_cache()
    return model, config


__all__ = ["load_model_from_checkpoint"]
