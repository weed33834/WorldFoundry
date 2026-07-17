"""Run the shared SHARP panorama-to-Gaussian inference model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.sharp import (
    build_panorama_predictor,
    load_predictor_checkpoint,
    save_panorama_ply,
)

LOGGER = logging.getLogger(__name__)


def _load_panorama(
    path: Path,
    height: int | None,
    width: int | None,
) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if height is None or width is None:
        height = max(16, round(image.height / 16) * 16)
        width = height * 2
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_depth(
    path: Path,
    height: int,
    width: int,
    factor: float,
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".png":
        depth = np.asarray(Image.open(path), dtype=np.float32) / 1000.0
    else:
        raise ValueError("SHARP depth input must be .npy or 16-bit .png.")
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}.")
    if depth.shape != (height, width):
        tensor = torch.from_numpy(depth)[None, None]
        depth = F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
    return np.clip(depth * factor, 0.001, None)


@torch.no_grad()
def infer(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_panorama_predictor(
        args.depth_model_path,
        num_layers=args.num_layers,
        device=device,
    )
    load_predictor_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    image = _load_panorama(args.image, args.pano_height, args.pano_width)
    height, width = image.shape[:2]
    depth = _load_depth(args.depth, height, width, args.depth_factor)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
    depth_tensor = torch.from_numpy(depth)[None, None].to(device)
    output = model(image_tensor, depth_gt=depth_tensor)

    destination = args.output_dir / f"{args.image.stem}.ply"
    save_panorama_ply(output.gaussians.to("cpu"), (height, width), destination)
    LOGGER.info("Saved %s", destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panorama RGBD to 3D Gaussian inference")
    parser.add_argument("--image", "-i", type=Path, required=True)
    parser.add_argument("--depth", "-d", type=Path, required=True)
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--depth_model_path", type=Path)
    parser.add_argument("--depth_factor", type=float, default=1.0)
    parser.add_argument("--pano_height", type=int)
    parser.add_argument("--pano_width", type=int)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    infer(parse_args())
