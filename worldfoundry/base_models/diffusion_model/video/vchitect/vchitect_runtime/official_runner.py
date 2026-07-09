"""Module for base_models -> diffusion_model -> video -> vchitect -> vchitect_runtime -> official_runner.py functionality."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Helper function to parse args.

    Returns:
        The return value.
    """
    parser = argparse.ArgumentParser(description="Run a bounded Vchitect-2 T2V validation.")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Helper function to validate args.

    Args:
        args: The args.

    Returns:
        The return value.
    """
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive.")
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("width and height must be divisible by 8.")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if args.fps <= 0:
        raise ValueError("fps must be positive.")


def _runtime_root() -> Path:
    """Helper function to runtime root.

    Returns:
        The return value.
    """
    return Path(__file__).resolve().parent


def _frames_to_video_array(frames: Any) -> np.ndarray:
    """Helper function to frames to video array.

    Args:
        frames: The frames.

    Returns:
        The return value.
    """
    arrays: list[np.ndarray] = []
    for frame in frames:
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
            if frame.ndim == 3 and frame.shape[0] in {1, 3, 4}:
                frame = np.transpose(frame, (1, 2, 0))
        elif hasattr(frame, "convert"):
            frame = np.asarray(frame.convert("RGB"))
        else:
            frame = np.asarray(frame)
        arrays.append(np.asarray(frame))
    if not arrays:
        raise ValueError("Vchitect runner returned no frames.")
    video = np.stack(arrays, axis=0)
    if video.dtype != np.uint8:
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(video * 255.0 if video.max() <= 1.0 else video, 0, 255)
        else:
            video = np.clip(video, 0, 255)
        video = video.astype(np.uint8)
    if video.shape[-1] == 4:
        video = video[..., :3]
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    return video


def _gpu_metadata() -> dict[str, Any]:
    """Helper function to gpu metadata.

    Returns:
        The return value.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu_device": torch.cuda.get_device_name(0),
        "max_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "torch_cuda": torch.version.cuda,
    }


def main() -> int:
    """Main.

    Returns:
        The return value.
    """
    started = time.monotonic()
    args = _parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Vchitect-2 official runner requires CUDA for GPU validation evidence.")

    from .models.pipeline import VchitectXLPipeline

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(0)
    load_started = time.monotonic()
    pipe = VchitectXLPipeline(str(checkpoint_root), device="cuda")
    load_seconds = time.monotonic() - load_started
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    generate_started = time.monotonic()
    frames = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        generator=generator,
    )
    generate_seconds = time.monotonic() - generate_started
    video = _frames_to_video_array(frames)
    imageio.mimsave(str(output_path), video, fps=args.fps)
    report = {
        "ok": True,
        "status": "success",
        "model_id": "vchitect-2-t2v",
        "artifact_kind": "generated_video",
        "output_path": str(output_path),
        "frame_count": int(video.shape[0]),
        "video_shape": list(video.shape),
        "video_dtype": str(video.dtype),
        "duration_seconds": round(time.monotonic() - started, 3),
        "load_seconds": round(load_seconds, 3),
        "generate_seconds": round(generate_seconds, 3),
        "torch_version": torch.__version__,
        "request": {
            "checkpoint_root": str(checkpoint_root),
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "fps": args.fps,
            "seed": args.seed,
        },
        **_gpu_metadata(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
