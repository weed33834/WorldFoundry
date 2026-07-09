"""Sharded safetensors loading helpers."""

from __future__ import annotations

import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from safetensors.torch import load as load_from_bytes
from safetensors.torch import load_file
from tqdm.auto import tqdm

from worldfoundry.core.distributed.logging import print_per_rank


def _load_shard(shard_path: str, param_names: list[str], num_threads: int | None = None):
    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        start_time = datetime.now()
        print_per_rank(f"Decompressing {zstd_path} with {num_threads} threads")
        cmd = ["zstd", "-d"]
        if num_threads:
            cmd.extend(["-T", str(num_threads)])

        process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
        decompressed_data = process.stdout.read()
        process.stdout.close()

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Decompression failed: {process.stderr.read().decode()}")
        print_per_rank(
            f"Decompressed {zstd_path} with {num_threads} threads, duration: {(datetime.now() - start_time).total_seconds()}s"
        )

        buffer = io.BytesIO(decompressed_data)
        start_time = datetime.now()
        print_per_rank(f"Loading {shard_path} from zstd file, start time: {start_time}")
        weights = load_from_bytes(buffer.getvalue())
        print_per_rank(f"Loaded {shard_path} from zstd file, duration: {(datetime.now() - start_time).total_seconds()}s")
        buffer.close()
    else:
        weights = load_file(shard_path)

    return {name: weights[name] for name in param_names}


def load_sharded_safetensors_parallel_with_progress(checkpoint_dir: str):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        return load_file(model_file_path)

    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    shard_map = {}
    for param_name, shard_file in index["weight_map"].items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        shard_map.setdefault(shard_path, []).append(param_name)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_load_shard, shard_path, param_names): shard_path
            for shard_path, param_names in shard_map.items()
        }
        pbar = tqdm(futures, desc="Loading shards", total=len(futures))
        for future in pbar:
            state_dict.update(future.result())

    return state_dict


def unwrap_model(model):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while hasattr(model_module, "module"):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


__all__ = ["load_sharded_safetensors_parallel_with_progress", "unwrap_model"]
