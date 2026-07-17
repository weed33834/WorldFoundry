"""Released checkpoint readers used by the Gamma text encoder."""

from pathlib import Path

import torch

from worldfoundry.core.io.easy_io import resolve_checkpoint_path

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".ckpt", ".pth", ".pt"}


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state = {}
    for path in sorted(Path(file_path).iterdir()):
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            state.update(load_state_dict(path, torch_dtype=torch_dtype))
    return state


def load_state_dict(file_path, torch_dtype=None):
    path = resolve_checkpoint_path(file_path)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        state = {
            key: value.to(torch_dtype) if isinstance(value, torch.Tensor) else value for key, value in state.items()
        }
    return state


__all__ = ["load_state_dict", "load_state_dict_from_folder"]
