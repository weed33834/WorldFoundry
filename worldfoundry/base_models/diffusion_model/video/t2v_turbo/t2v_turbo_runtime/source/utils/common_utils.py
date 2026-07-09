"""Small inference-only subset of the official T2V-Turbo common utilities."""

from __future__ import annotations

import gc

import torch


def load_model_checkpoint(model, ckpt: str):
    """Load a full official T2V-Turbo / VideoCrafter checkpoint."""

    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = {"scale_arr_prev"}
    unexpected_keys = set(unexpected)
    missing_keys = set(missing) - ignored_missing
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Unexpected T2V-Turbo checkpoint mismatch: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
        )
    if missing:
        print(f">>> model checkpoint loaded with initialized buffers: {sorted(missing)}")
    else:
        print(">>> model checkpoint loaded.")
    del state_dict
    gc.collect()
    return model
