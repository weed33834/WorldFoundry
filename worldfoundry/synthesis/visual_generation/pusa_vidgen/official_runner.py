from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Pusa VidGen Genmo/Mochi validation.")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-type", default="full", choices=("full", "tiled_spatial", "tiled_full"))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive.")
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("width and height must be divisible by 8 for Mochi latents.")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if (args.num_frames - 1) % 6 != 0:
        raise ValueError("num_frames - 1 must be divisible by 6.")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")


def _checkpoint_file(root: Path, filename: str) -> str:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"required Pusa checkpoint file is missing: {path}")
    return str(path)


class SplitT5ModelFactory:
    """Load tokenizer and T5 weights from the split Pusa checkpoint layout."""

    def __init__(self, *, text_encoder_dir: Path, tokenizer_dir: Path) -> None:
        self.text_encoder_dir = text_encoder_dir
        self.model_dir = str(tokenizer_dir)

    def get_model(self, *, local_rank: int, device_id: int | str, world_size: int) -> Any:
        del local_rank
        if world_size != 1:
            raise ValueError("SplitT5ModelFactory only supports single-GPU Pusa validation runs.")
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(str(self.text_encoder_dir), local_files_only=True)
        if isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))
        return model.eval()


def _gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu_device": torch.cuda.get_device_name(0),
        "max_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "torch_cuda": torch.version.cuda,
    }


def _sigma_schedule(num_inference_steps: int) -> list[float]:
    """Return a valid Mochi sigma schedule, including the one-step validation case."""
    if num_inference_steps == 1:
        return [1.0, 0.0]
    from genmo.mochi_preview.pipelines import linear_quadratic_schedule

    return list(linear_quadratic_schedule(num_inference_steps, 0.025))


def _install_optional_ray_stub() -> None:
    try:
        import ray  # noqa: F401
        return
    except ImportError:
        pass

    ray_stub = types.ModuleType("ray")

    def _missing_ray(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ImportError("ray is required for Pusa multi-GPU mode, but this runner uses single-GPU validation inference.")

    ray_stub.init = _missing_ray  # type: ignore[attr-defined]
    ray_stub.remote = _missing_ray  # type: ignore[attr-defined]
    ray_stub.get = _missing_ray  # type: ignore[attr-defined]
    sys.modules.setdefault("ray", ray_stub)


def _save_video_compat(frames: Any, output_path: Path, *, fps: int = 30) -> None:
    from genmo.lib.utils import save_video

    try:
        save_video(frames, str(output_path), fps=fps)
        return
    except TypeError:
        pass

    import imageio.v3 as iio

    array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    iio.imwrite(str(output_path), array, fps=fps)


def main() -> int:
    started = time.monotonic()
    args = _parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Pusa VidGen official runner requires CUDA for open-source GPU validation evidence.")

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from genmo.lib.progress import progress_bar
    _install_optional_ray_stub()
    from genmo.mochi_preview.pipelines import (
        DecoderModelFactory,
        DitModelFactory,
        MochiSingleGPUPipeline,
    )

    pipeline = MochiSingleGPUPipeline(
        text_encoder_factory=SplitT5ModelFactory(
            text_encoder_dir=checkpoint_root / "text_encoder",
            tokenizer_dir=checkpoint_root / "tokenizer",
        ),
        dit_factory=DitModelFactory(
            model_path=_checkpoint_file(checkpoint_root, "pusa_v0_dit.safetensors"),
            model_dtype="bf16",
            attention_mode="sdpa",
        ),
        decoder_factory=DecoderModelFactory(
            model_path=_checkpoint_file(checkpoint_root, "decoder.safetensors"),
        ),
        cpu_offload=bool(args.cpu_offload),
        decode_type=args.decode_type,
        decode_args={"overlap": 8} if args.decode_type.startswith("tiled_") else {},
        fast_init=True,
        strict_load=True,
    )
    sigma_schedule = _sigma_schedule(args.num_inference_steps)
    cfg_schedule = [args.cfg_scale] * args.num_inference_steps
    with progress_bar(type="tqdm"):
        frames = pipeline(
            batch_cfg=False,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            sigma_schedule=sigma_schedule,
            cfg_schedule=cfg_schedule,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )
    _save_video_compat(frames[0], output_path, fps=30)
    report = {
        "ok": True,
        "status": "success",
        "artifact_kind": "generated_video",
        "output_path": str(output_path),
        "frames_shape": list(frames.shape),
        "frames_dtype": str(frames.dtype),
        "duration_seconds": round(time.monotonic() - started, 3),
        "torch_version": torch.__version__,
        "request": {
            "checkpoint_root": str(checkpoint_root),
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "cpu_offload": bool(args.cpu_offload),
            "decode_type": args.decode_type,
        },
        **_gpu_metadata(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
