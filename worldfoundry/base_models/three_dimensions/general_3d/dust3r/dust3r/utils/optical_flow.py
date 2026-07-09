"""Module for base_models -> three_dimensions -> general_3d -> dust3r -> dust3r -> utils -> optical_flow.py functionality."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from worldfoundry.base_models.perception_core.optical_flow.raft.raft import RAFT as ClassicRAFT
from worldfoundry.base_models.perception_core.optical_flow.sea_raft.core.raft import RAFT as SeaRAFT


def _find_ckpt_root():
    """Helper function to find ckpt root."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "ckpt"
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ckpt(model_path):
    """Helper function to resolve ckpt.

    Args:
        model_path: The model path.
    """
    ckpt_root = _find_ckpt_root()
    if model_path is None:
        model_path = "raft-things.pth"

    model_name = Path(model_path).name
    if ckpt_root is not None:
        for directory in (
            ckpt_root / "monst3r_raft_models",
            ckpt_root / "WorldScore",
            ckpt_root / "WorldScore" / "models",
        ):
            candidate = directory / model_name
            if candidate.exists():
                return candidate

    path = Path(model_path)
    if path.exists():
        return path
    raise FileNotFoundError(f"RAFT checkpoint not found: {model_path}")


def _clean_state_dict(state_dict):
    """Helper function to clean state dict.

    Args:
        state_dict: The state dict.
    """
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_classic_raft(ckpt_path):
    """Helper function to load classic raft.

    Args:
        ckpt_path: The ckpt path.
    """
    args = argparse.Namespace(
        model=str(ckpt_path),
        path="./",
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0,
    )
    model = ClassicRAFT(args)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(_clean_state_dict(state_dict), strict=False)
    return model.eval()


class _SeaRAFTCompat(nn.Module):
    """Sea raft compat implementation."""
    def __init__(self, model):
        """Init.

        Args:
            model: The model.
        """
        super().__init__()
        self.model = model

    def forward(self, image1, image2, iters=20, test_mode=False):
        """Forward.

        Args:
            image1: The image1.
            image2: The image2.
            iters: The iters.
            test_mode: The test mode.
        """
        output = self.model(image1, image2, iters=iters, test_mode=True)
        flow = output["final"]
        return None, flow


def _load_sea_raft(ckpt_path):
    """Helper function to load sea raft.

    Args:
        ckpt_path: The ckpt path.
    """
    args = argparse.Namespace(
        use_var=True,
        var_min=0,
        var_max=10,
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256],
        radius=4,
        dim=128,
        num_blocks=2,
        num_head=1,
        iters=4,
        scale=-1,
    )
    model = SeaRAFT(args)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return _SeaRAFTCompat(model).eval()


def load_raft(model_path=None):
    """Load raft.

    Args:
        model_path: The model path.
    """
    ckpt_path = _resolve_ckpt(model_path)
    if "M" in ckpt_path.name:
        return _load_sea_raft(ckpt_path)
    return _load_classic_raft(ckpt_path)
