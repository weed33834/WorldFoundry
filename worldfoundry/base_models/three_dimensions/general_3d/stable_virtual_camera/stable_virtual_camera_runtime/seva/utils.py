"""Module for base_models -> three_dimensions -> general_3d -> stable_virtual_camera -> stable_virtual_camera_runtime -> seva -> utils.py functionality."""

import os

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from seva.model import Seva, SevaParams


def seed_everything(seed: int = 0):
    """Seed everything.

    Args:
        seed: The seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    """Print load warning.

    Args:
        missing: The missing.
        unexpected: The unexpected.

    Returns:
        The return value.
    """
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_model(
    model_version: float = 1.1,
    pretrained_model_name_or_path: str = "stabilityai/stable-virtual-camera",
    weight_name: str = "model.safetensors",
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> Seva:
    """Load model.

    Args:
        model_version: The model version.
        pretrained_model_name_or_path: The pretrained model name or path.
        weight_name: The weight name.
        device: The device.
        verbose: The verbose.

    Returns:
        The return value.
    """
    if os.path.isdir(pretrained_model_name_or_path):
        weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
    else:
        if model_version > 1:
            base, ext = os.path.splitext(weight_name)
            weight_name = f"{base}v{model_version}{ext}"
        weight_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=weight_name
        )
        _ = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="config.yaml"
        )

    state_dict = safetensors.torch.load_file(
        weight_path,
        device=str(device),
    )

    with torch.device("meta"):
        model = Seva(SevaParams()).to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)
    return model
