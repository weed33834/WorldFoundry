"""Inference entrypoint for the in-tree Gamma-World runtime."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from worldfoundry.core.io import action_track_tensors, format_override_value
from worldfoundry.core.utils import split_horizontal_views

LOGGER = logging.getLogger("worldfoundry.gamma_world")
MODES = ("bidirectional", "causal", "causal_few_step")


def _optional_path(value: str) -> str | None:
    return value or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamma-World inference through WorldFoundry")
    parser.add_argument("--mode", choices=MODES, default="causal_few_step")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--vae", default="")
    parser.add_argument("--text-encoder", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.n_players <= 0:
        parser.error("--n-players must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    return args


def _resolve_sample_dir(eval_dir: str | Path) -> Path:
    root = Path(eval_dir).expanduser().resolve()
    if (root / "first_frame.png").is_file():
        return root
    samples = sorted(path.parent for path in root.rglob("first_frame.png"))
    if not samples:
        raise FileNotFoundError(f"no first_frame.png found under {root}")
    if len(samples) > 1:
        raise ValueError(
            f"WorldFoundry's single-output Gamma runner requires one sample; found {len(samples)} under {root}"
        )
    return samples[0]


def _action_paths(sample_dir: Path, n_players: int) -> list[Path] | None:
    paths = [sample_dir / f"action_{index}.json" for index in range(n_players)]
    if all(path.is_file() for path in paths):
        return paths
    if n_players == 2:
        legacy = [sample_dir / "action_left.json", sample_dir / "action_right.json"]
        if all(path.is_file() for path in legacy):
            return legacy
    return None


def _load_sample(args: argparse.Namespace):
    sample_dir = _resolve_sample_dir(args.eval_dir)
    images = [
        np.asarray(image)
        for image in split_horizontal_views(
            sample_dir / "first_frame.png",
            num_views=args.n_players,
            target_size=(args.width, args.height),
        )
    ]
    if args.prompt:
        prompt = args.prompt
    else:
        prompt_path = sample_dir / "prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"{prompt_path} not found and --prompt is empty")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    paths = _action_paths(sample_dir, args.n_players)
    actions = [action_track_tensors(path, num_frames=args.num_frames) for path in paths] if paths is not None else None
    return images, prompt, actions


def main() -> None:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    import torch

    from worldfoundry.synthesis.visual_generation.gamma_world.bidirectional_inference import (
        BidirectionalInference,
    )
    from worldfoundry.synthesis.visual_generation.gamma_world.causal_inference import I2VInference
    from worldfoundry.synthesis.visual_generation.gamma_world.model_specs import MODEL_SPECS

    torch.set_grad_enabled(False)
    spec = MODEL_SPECS[args.mode]
    checkpoint = _optional_path(args.checkpoint) or spec.default_checkpoint
    num_steps = args.num_steps if args.num_steps > 0 else spec.default_num_steps
    experiment_opts = [f"{key}={format_override_value(value)}" for key, value in spec.config_overrides.items()]
    engine_cls = BidirectionalInference if spec.sampler == "bidirectional" else I2VInference
    engine = engine_cls(
        experiment_name=spec.experiment,
        ckpt_path=checkpoint,
        config_file=spec.config_file,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_steps=num_steps or 35,
        seed=args.seed,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=experiment_opts,
        vae_pth=_optional_path(args.vae),
        text_encoder_pth=_optional_path(args.text_encoder),
    )
    engine.fps = args.fps
    images, prompt, actions = _load_sample(args)
    batch = engine.build_inference_batch(
        images,
        prompt,
        actions,
        num_frames=args.num_frames,
        num_conditional_frames=args.num_conditional_frames,
    )
    engine.clear_cache()
    generate_kwargs = {
        "guidance": args.guidance,
        "seed": args.seed,
        "num_steps": num_steps,
        "shift": args.shift,
    }
    if spec.sampler == "bidirectional":
        video = engine.generate_single_shot(batch, **generate_kwargs)
    else:
        video = engine.generate_from_batch(batch, **generate_kwargs)
    if engine.rank0:
        from worldfoundry.core.io import save_image_or_video_tensor

        video = ((video + 1.0) / 2.0).clamp(0, 1)
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = save_image_or_video_tensor(video[0], output_path, fps=args.fps, value_range=(0.0, 1.0))
        LOGGER.info("saved %s to %s", tuple(video.shape), saved_path)


if __name__ == "__main__":
    main()
