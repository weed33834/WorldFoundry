"""State-dict loading from safetensors, bins, folders, and remote URIs."""

import hashlib
import json
import os
import pickle
from typing import Any

import torch
from safetensors import safe_open

from worldfoundry.core.io.storage import (
    is_dir_uri,
    is_file_uri,
    join_uri,
    list_uri,
    local_path_for_uri,
    parse_uri_scheme,
    read_text_uri,
)


def load_state_dict(file_path, torch_dtype=None, device="cpu", pin_memory=False, verbose=0):
    """Load a checkpoint file, folder, or sharded safetensors index into a state dict."""
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device, pin_memory=pin_memory, verbose=verbose))
    else:
        file_path = str(file_path)
        if verbose >= 1:
            print(f"Loading file [started]: {file_path}")
        if is_dir_uri(file_path):
            state_dict = load_state_dict_from_folder(file_path, torch_dtype=torch_dtype, device=device, pin_memory=False, verbose=verbose)
        elif file_path.endswith(".safetensors.index.json"):
            state_dict = load_state_dict_from_safetensors_index(file_path, torch_dtype=torch_dtype, device=device)
        elif file_path.endswith(".safetensors"):
            state_dict = load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
        else:
            state_dict = load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)
        # If load state dict in CPU memory, `pin_memory=True` will make `model.to("cuda")` faster.
        if pin_memory:
            for i in state_dict:
                state_dict[i] = state_dict[i].pin_memory()
        if verbose >= 1:
            print(f"Loading file [done]: {file_path}")
    return state_dict


def load_state_dict_from_folder(file_path, torch_dtype=None, device="cpu", pin_memory=False, verbose=0):
    file_path = str(file_path)
    if parse_uri_scheme(file_path) == "file":
        file_names = [join_uri(file_path, file_name) for file_name in os.listdir(file_path)]
    else:
        file_names = list_uri(file_path)
    index_files = sorted(file_name for file_name in file_names if file_name.endswith(".safetensors.index.json"))
    if index_files:
        state_dict = {}
        for file_path_ in index_files:
            state_dict.update(
                load_state_dict_from_safetensors_index(
                    file_path_,
                    torch_dtype=torch_dtype,
                    device=device,
                )
            )
        if pin_memory:
            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    state_dict[name] = tensor.pin_memory()
        return state_dict
    state_dict = {}
    for file_path_ in file_names:
        file_name = os.path.basename(file_path_)
        if "." not in file_name:
            continue
        if file_name.rsplit(".", 1)[-1] not in {"safetensors", "bin", "ckpt", "pth", "pt"}:
            continue
        state_dict.update(
            load_state_dict(
                file_path_,
                torch_dtype=torch_dtype,
                device=device,
                pin_memory=pin_memory,
                verbose=verbose,
            )
        )
    return state_dict


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with local_path_for_uri(file_path) as local_path:
        with safe_open(str(local_path), framework="pt", device=str(device)) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
                if torch_dtype is not None:
                    state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_safetensors_index(file_path, torch_dtype=None, device="cpu"):
    file_path = str(file_path)
    index = json.loads(read_text_uri(file_path))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Safetensors index {file_path!r} does not contain a weight_map")
    root = file_path.rsplit("/", 1)[0]
    state_dict = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = join_uri(root, shard_name)
        state_dict.update(load_state_dict_from_safetensors(shard_path, torch_dtype=torch_dtype, device=device))
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = load_torch_state_dict(file_path, map_location=device)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def _torch_load(checkpoint_path: str | os.PathLike[str], **kwargs: Any) -> Any:
    with local_path_for_uri(checkpoint_path) as local_path:
        try:
            return torch.load(str(local_path), **kwargs)
        except TypeError as exc:
            if "weights_only" not in str(exc) or "weights_only" not in kwargs:
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("weights_only")
            return torch.load(str(local_path), **fallback_kwargs)


def load_torch_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
    weights_only: bool | None = True,
    allow_unsafe_pickle_fallback: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a PyTorch checkpoint with a safe default and old-PyTorch fallback."""
    load_kwargs = {"map_location": map_location, **kwargs}
    if weights_only is not None:
        load_kwargs["weights_only"] = weights_only
    try:
        return _torch_load(checkpoint_path, **load_kwargs)
    except pickle.UnpicklingError as exc:
        if not (weights_only and allow_unsafe_pickle_fallback and "Weights only load failed" in str(exc)):
            raise
        fallback_kwargs = {"map_location": map_location, "weights_only": False, **kwargs}
        return _torch_load(checkpoint_path, **fallback_kwargs)


def load_torch_state_dict(checkpoint_path: str | os.PathLike[str], *, map_location: Any = "cpu") -> Any:
    """Load PyTorch state dictionaries through weights-only deserialization when available."""
    return load_torch_checkpoint(checkpoint_path, map_location=map_location, weights_only=True)


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_state_dict_keys(state_dict, with_shape=True):
    """Return an MD5 digest of state-dict key names (optionally including shapes)."""
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


def split_state_dict_with_prefix(state_dict):
    """Split a state dict into sub-dicts grouped by the first dotted key segment."""

    prefix_dict = {}
    for key in sorted(key for key in state_dict if isinstance(key, str)):
        prefix = key if "." not in key else key.split(".", 1)[0]
        prefix_dict.setdefault(prefix, []).append(key)
    return [{key: state_dict[key] for key in keys} for keys in prefix_dict.values()]


def search_for_embeddings(state_dict):
    """Return all tensor leaves from a nested state dict."""

    embeddings = []
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            embeddings.append(value)
        elif isinstance(value, dict):
            embeddings.extend(search_for_embeddings(value))
    return embeddings


def search_parameter(param, state_dict, *, atol=1e-3):
    """Find the first state-dict key whose tensor numerically matches ``param``."""

    for name, candidate in state_dict.items():
        if not isinstance(candidate, torch.Tensor) or param.numel() != candidate.numel():
            continue
        if param.shape == candidate.shape:
            distance = torch.dist(param, candidate)
        else:
            distance = torch.dist(param.flatten(), candidate.flatten())
        if distance < atol:
            return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    """Print parameter-key matches between two state dicts for conversion scripts."""

    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = [
                    search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    for i in range(3)
                ]
                if None not in rename:
                    print(f'"{name}": {rename},')
                    matched_keys.update(rename)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    """Recursively find files matching any suffix in ``extensions``."""

    suffixes = tuple(extensions)
    if is_dir_uri(folder):
        return list_uri(folder, recursive=True, suffix=suffixes)
    if is_file_uri(folder) and str(folder).endswith(suffixes):
        return [str(folder)]
    return []


def load_keys_dict(file_path):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_keys_dict(file_path_))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_keys_dict_from_safetensors(file_path)
    else:
        return load_keys_dict_from_bin(file_path)


def load_keys_dict_from_safetensors(file_path):
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            keys_dict[k] = f.get_slice(k).get_shape()
    return keys_dict


def convert_state_dict_to_keys_dict(state_dict):
    keys_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            keys_dict[k] = list(v.shape)
        else:
            keys_dict[k] = convert_state_dict_to_keys_dict(v)
    return keys_dict


def load_keys_dict_from_bin(file_path):
    state_dict = load_state_dict_from_bin(file_path)
    keys_dict = convert_state_dict_to_keys_dict(state_dict)
    return keys_dict


def convert_keys_dict_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_model_file(path, with_shape=True):
    """Return an MD5 digest of checkpoint key names loaded from *path*."""
    keys_dict = load_keys_dict(path)
    keys_str = convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


__all__ = [
    "convert_keys_dict_to_single_str",
    "convert_state_dict_keys_to_single_str",
    "convert_state_dict_to_keys_dict",
    "hash_model_file",
    "hash_state_dict_keys",
    "load_keys_dict",
    "load_keys_dict_from_bin",
    "load_keys_dict_from_safetensors",
    "load_state_dict",
    "load_state_dict_from_bin",
    "load_state_dict_from_folder",
    "load_state_dict_from_safetensors",
    "load_torch_checkpoint",
    "load_torch_state_dict",
    "build_rename_dict",
    "search_for_embeddings",
    "search_for_files",
    "search_parameter",
    "split_state_dict_with_prefix",
]
