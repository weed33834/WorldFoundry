# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> wan -> wan_runtime_wrapper.py functionality."""

import copy
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
import yaml
from PIL import Image

from worldfoundry.evaluation.utils import DATA_ROOT
from worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.configs import (
    MAX_AREA_CONFIGS,
    SIZE_CONFIGS,
    SUPPORTED_SIZES,
    WAN_CONFIGS,
)

_WAN_EXAMPLE_CONFIG_PATH = DATA_ROOT / "models" / "runtime" / "configs" / "wan" / "prompt_defaults.yaml"
_WAN_EXAMPLE_ASSET_ROOT = DATA_ROOT / "test_cases"
_WAN_EXAMPLE_PROMPT_FALLBACK = {
    "t2v-1.3B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "wan/i2v_input.png",
    },
    "flf2v-14B": {
        "prompt":
            "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
        "first_frame":
            "wan/flf2v_input_first_frame.png",
        "last_frame":
            "wan/flf2v_input_last_frame.png",
    },
    "vace-1.3B": {
        "src_ref_images":
            "wan/girl.png,wan/snake.png",
        "prompt":
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年气氛。"
    },
    "vace-14B": {
        "src_ref_images":
            "wan/girl.png,wan/snake.png",
        "prompt":
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年气氛。"
    },
}


def _normalize_prompt_config(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Helper function to normalize prompt config.

    Args:
        payload: The payload.

    Returns:
        The return value.
    """
    prompts = payload.get("tasks")
    if not isinstance(prompts, dict):
        prompts = _WAN_EXAMPLE_PROMPT_FALLBACK
    return copy.deepcopy(prompts)


def _resolve_wan_data_path(value: Any) -> Any:
    """Helper function to resolve wan data path.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    if not isinstance(value, str):
        return value
    path_value = value.strip()
    if not path_value:
        return value
    if "," in path_value:
        chunks = [chunk.strip() for chunk in path_value.split(",") if chunk.strip()]
        return ",".join(_resolve_wan_data_path(chunk) for chunk in chunks)
    path = Path(path_value)
    if not path.is_absolute():
        path = _WAN_EXAMPLE_ASSET_ROOT / path
    return str(path) if path.exists() else value


def _load_example_prompts() -> dict[str, dict[str, str]]:
    """Helper function to load example prompts.

    Returns:
        The return value.
    """
    payload: dict[str, Any] = {}
    try:
        payload_raw = yaml.safe_load(_WAN_EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(payload_raw, dict):
            payload = payload_raw
    except (OSError, yaml.YAMLError):
        payload = {}

    prompts = _normalize_prompt_config(payload)
    media_fields = {"image", "first_frame", "last_frame", "src_ref_images"}
    for task_values in prompts.values():
        if not isinstance(task_values, dict):
            continue
        for field in media_fields:
            if field in task_values:
                task_values[field] = _resolve_wan_data_path(task_values[field])
    return prompts


EXAMPLE_PROMPT = _load_example_prompts()


def _init_logging(rank):
    """Helper function to init logging.

    Args:
        rank: The rank.
    """
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

class Wan:
  """Wan implementation."""
  def __init__(
    self,
    model_name: str,
    generation_type: Literal["t2v", "i2v"],
    offload_model: bool = None,
    t5_fsdp: bool = False,
    dit_fsdp: bool = False,
    ulysses_size: int = 1,
    ring_size: int = 1,
    use_prompt_extend: bool = False,
    prompt_extend_method: str = "local_qwen",
    prompt_extend_model: str = None,
    prompt_extend_target_lang: str = "zh",
    sample_steps: Optional[int] = None,
    sample_shift: Optional[float] = None,
    base_seed: int = -1,
    sample_solver: str = "unipc",
    sample_guide_scale: float = 5.0,
    task: str = "i2v-14B",
    size: str = "832*480",
    ckpt_dir: str = "./models/Wan2.1-I2V-14B-480P",
    frames: int = 81,
    fps: int = 16,
    t5_cpu: bool = False,
  ):
    """Create an in-tree Wan2.1 local runtime.

    Args:
      model_name: Model id used for reporting.
      generation_type: Selects the text-to-video or image-to-video pipeline.
      ckpt_dir: External checkpoint directory; code stays under worldfoundry.
    """
    self.generation_type = generation_type
    
    frame_num = frames
    base_seed = base_seed if base_seed >= 0 else random.randint(
        0, sys.maxsize)
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if offload_model is None:
        offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            t5_fsdp or dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            ulysses_size > 1 or ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if ulysses_size > 1 or ring_size > 1:
        assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=ring_size,
            ulysses_degree=ulysses_size,
        )

    if use_prompt_extend:
        from worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.prompt_extend import (
            DashScopePromptExpander,
            QwenPromptExpander,
        )

        if prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=prompt_extend_model,
                is_vl="i2v" in task or "flf2v" in task)
        elif prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=prompt_extend_model,
                is_vl="i2v" in task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {prompt_extend_method}")

    cfg = WAN_CONFIGS[task]
    if task not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported Wan task: {task}")
    if size not in SUPPORTED_SIZES[task]:
        raise ValueError(f"Unsupported size {size!r} for task {task}; supported sizes: {SUPPORTED_SIZES[task]}")
    if sample_steps is None:
        sample_steps = 40 if "i2v" in task else 50
    if sample_shift is None:
        sample_shift = 5.0
        if "i2v" in task and size in ["832*480", "480*832"]:
            sample_shift = 3.0
        elif "flf2v" in task or "vace" in task:
            sample_shift = 16.0
    if ulysses_size > 1:
        assert cfg.num_heads % ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{ulysses_size=}`."

    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        base_seed = base_seed[0]

    
    self.wan_t2v = None
    self.wan_i2v = None
    common_kwargs = {
        "config": cfg,
        "checkpoint_dir": ckpt_dir,
        "device_id": device,
        "rank": rank,
        "t5_fsdp": t5_fsdp,
        "dit_fsdp": dit_fsdp,
        "use_usp": (ulysses_size > 1 or ring_size > 1),
        "t5_cpu": t5_cpu,
    }
    if generation_type == "t2v":
        from worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.text2video import WanT2V

        logging.info("Creating WanT2V pipeline.")
        self.wan_t2v = WanT2V(**common_kwargs)
    elif generation_type == "i2v":
        from worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.image2video import WanI2V

        logging.info("Creating WanI2V pipeline.")
        self.wan_i2v = WanI2V(**common_kwargs)
    else:
        raise ValueError(f"Unsupported Wan generation_type: {generation_type!r}")
    
    self.frame_num = frame_num
    self.sample_shift = sample_shift
    self.sample_solver = sample_solver
    self.sample_steps = sample_steps
    self.sample_guide_scale = sample_guide_scale
    self.base_seed = base_seed
    self.offload_model = offload_model
    self.size = size
    self.fps = fps

    
  def generate_video(
  	self,
    prompt: str,
    image_path: Optional[str] = None,
  ):
    """Generate video frames from a prompt and optional image path.

    Args:
      prompt: Text prompt passed to the Wan pipeline.
      image_path: Required source image path for image-to-video generation.
    """
    logging.info("Generating video ...")
    if self.generation_type == "t2v":
        if self.wan_t2v is None:
            raise RuntimeError("WanT2V pipeline is not initialized.")
        video = self.wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[self.size],
            frame_num=self.frame_num,
            shift=self.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.sample_steps,
            guide_scale=self.sample_guide_scale,
            seed=self.base_seed,
            offload_model=self.offload_model)
    else:
        if self.wan_i2v is None:
            raise RuntimeError("WanI2V pipeline is not initialized.")
        if image_path is None:
            raise ValueError("Wan image-to-video generation requires image_path.")
        img = Image.open(image_path).convert("RGB")
        video = self.wan_i2v.generate(
            prompt,
            img,
            max_area=MAX_AREA_CONFIGS[self.size],
            frame_num=self.frame_num,
            shift=self.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.sample_steps,
            guide_scale=self.sample_guide_scale,
            seed=self.base_seed,
            offload_model=self.offload_model)
    
    video = video.permute(1, 0, 2, 3)
    video = (video + 1.0) / 2.0
    # Must return either: 
    # - List[Image.Image], or 
    # - torch.Tensor of shape [N, 3, H, W] with values in [0, 1]
    return video
