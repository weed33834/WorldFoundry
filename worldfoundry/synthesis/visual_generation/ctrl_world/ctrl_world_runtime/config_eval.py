from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import CTRL_WORLD_DATASET_ROOT, CTRL_WORLD_META_INFO_ROOT


def _dataset_path(*parts: str) -> str:
    return str(CTRL_WORLD_DATASET_ROOT.joinpath(*parts))


def _meta_info_path(*parts: str) -> str:
    return str(CTRL_WORLD_META_INFO_ROOT.joinpath(*parts))


@dataclass
class wm_args:
    """Open-source eval/inference defaults for the in-tree Ctrl-World runtime."""

    task_type: str = "pickplace"
    svd_model_path: str = "stabilityai/stable-video-diffusion-img2vid"
    clip_model_path: str = "openai/clip-vit-base-patch32"
    ckpt_path: str = "yjguo/Ctrl-World"
    pi_ckpt: str = ""
    dataset_root_path: str = str(CTRL_WORLD_DATASET_ROOT)
    dataset_names: str = "droid_subset"
    dataset_meta_info_path: str = str(CTRL_WORLD_META_INFO_ROOT)
    dataset_cfgs: str = "droid_subset"
    prob: list[float] = field(default_factory=lambda: [1.0])
    annotation_name: str = "annotation"
    num_workers: int = 4
    down_sample: int = 3
    skip_step: int = 1
    debug: bool = False
    output_dir: str = "outputs/ctrl_world"
    motion_bucket_id: int = 127
    fps: int = 7
    guidance_scale: float = 1.0
    num_inference_steps: int = 50
    decode_chunk_size: int = 7
    width: int = 320
    height: int = 192
    num_frames: int = 5
    num_history: int = 6
    action_dim: int = 7
    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    dtype: torch.dtype = torch.bfloat16
    policy_type: str = "pi05"
    action_adapter: str = "models/action_adapter/model2_15_9.pth"
    pred_step: int = 5
    policy_skip_step: int = 2
    interact_num: int = 12
    data_stat_path: str = _meta_info_path("droid", "stat.json")
    history_idx: list[int] = field(default_factory=lambda: [0, 0, -12, -9, -6, -3])
    save_dir: str = "synthetic_traj"

    def __post_init__(self) -> None:
        self.dataset_cfgs = self.dataset_names
        if self.task_type == "replay":
            self.val_dataset_dir = _dataset_path("droid_subset")
            self.val_id = ["899", "18599", "199"]
            self.start_idx = [8, 14, 8]
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"
            return

        if self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = _dataset_path("droid_new_setup_full", "pickplace")
            self.val_id = ["0000"]
            self.start_idx = [0]
            self.instruction = ["pick up the blue block and place in white plate"]
            self.task_name = "Rollouts_interact_pi"
            return

        task_root = CTRL_WORLD_DATASET_ROOT / "droid_new_setup_full" / self.task_type
        self.val_dataset_dir = str(task_root)
        self.val_id = []
        self.start_idx = []
        self.instruction = []
        self.task_name = f"Rollouts_{self.task_type}"

