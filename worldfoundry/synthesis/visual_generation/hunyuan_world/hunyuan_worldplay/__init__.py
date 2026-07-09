import os
import socket
from importlib import import_module
from typing import Any

from .commons import get_gpu_memory

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _init_dist_env():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(find_free_port())


_init_dist_env()

__all__ = [
    "HunyuanVideoPipelineOutput",
    "HunyuanWorldPlayRuntime",
    "_HunyuanWorldPlayInternalPipeline",
    "find_free_port",
    "get_gpu_memory",
    "load_runtime",
]


def __getattr__(name: str) -> Any:
    if name in {
        "HunyuanVideoPipelineOutput",
        "HunyuanWorldPlayRuntime",
        "_HunyuanWorldPlayInternalPipeline",
        "load_runtime",
    }:
        module = import_module(".runtime", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
