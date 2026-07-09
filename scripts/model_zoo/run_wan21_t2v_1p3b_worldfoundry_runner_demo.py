#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
DEFAULT_CKPT_DIR = (
    REPO_ROOT
    / "cache"
    / "hfd"
    / "models--Wan-AI--Wan2.1-T2V-1.3B"
    / "snapshots"
    / "37ec512624d61f7aa208f7ea8140a131f93afc9a"
)
DEFAULT_OUTPUT_NAME = "wan21_t2v_1p3b_worldfoundry_seed42_832x480_81f.mp4"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR",
            str(REPO_ROOT / "tmp" / "model_zoo" / "parity" / "wan2.1"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WorldFoundry vendored Wan2.1 T2V-1.3B runner demo used for official parity checks."
    )
    parser.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("WAN21_CKPT_DIR", DEFAULT_CKPT_DIR)))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=8.0)
    parser.add_argument("--sample-guide-scale", type=float, default=6.0)
    parser.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--offload-model", type=str_to_bool, default=True)
    parser.add_argument("--t5-cpu", type=str_to_bool, default=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser


def init_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def write_metadata(path: Path, *, args: argparse.Namespace, output_path: Path, rank: int) -> None:
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "worldfoundry-wan21-runner-demo",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.WanT2V",
        "official_reference": "Wan-Video/Wan2.1 generate.py t2v-1.3B",
        "prompt": args.prompt,
        "task": "t2v-1.3B",
        "size": args.size,
        "frame_num": args.frame_num,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "sample_solver": args.sample_solver,
        "offload_model": args.offload_model,
        "t5_cpu": args.t5_cpu,
        "ckpt_dir": str(args.ckpt_dir),
        "output": str(output_path),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    wan = importlib.import_module("worldfoundry.base_models.diffusion_model.video.wan.wan_2p1")
    configs = importlib.import_module("worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.configs")
    media_utils = importlib.import_module(
        "worldfoundry.base_models.diffusion_model.video.wan.shared.media_utils"
    )
    cache_video = media_utils.cache_video
    size_configs = configs.SIZE_CONFIGS
    supported_sizes = configs.SUPPORTED_SIZES
    wan_configs = configs.WAN_CONFIGS

    if not args.ckpt_dir.is_dir():
        print(f"missing Wan2.1 T2V-1.3B checkpoint directory: {args.ckpt_dir}", file=sys.stderr)
        return 4
    if args.size not in supported_sizes["t2v-1.3B"]:
        print(
            f"unsupported size {args.size!r}; supported: {', '.join(supported_sizes['t2v-1.3B'])}",
            file=sys.stderr,
        )
        return 2

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    init_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    output_path = args.save_file or (args.output_dir / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_json or output_path.with_suffix(".json")

    cfg = wan_configs["t2v-1.3B"]
    logging.info("Creating WorldFoundry WanT2V runner pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=local_rank,
        rank=rank,
        t5_cpu=args.t5_cpu,
    )
    logging.info("Generating Wan2.1 T2V runner parity video.")
    video = wan_t2v.generate(
        args.prompt,
        size=size_configs[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    if rank == 0:
        logging.info("Saving generated video to %s", output_path)
        saved = cache_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if saved is None:
            return 5
        write_metadata(metadata_path, args=args, output_path=output_path, rank=rank)
        logging.info("Finished.")

    if dist.is_initialized():
        dist.barrier()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
