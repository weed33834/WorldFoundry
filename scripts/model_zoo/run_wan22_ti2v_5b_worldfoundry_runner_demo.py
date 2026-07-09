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
    / "models--Wan-AI--Wan2.2-TI2V-5B"
    / "snapshots"
    / "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
)
DEFAULT_OUTPUT_NAME = "wan22_ti2v_5b_worldfoundry_seed42_1280x704_121f.mp4"


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
            str(REPO_ROOT / "tmp" / "model_zoo" / "parity" / "wan2.2"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WorldFoundry Wan2.2 TI2V-5B runner demo used for official parity checks."
    )
    parser.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("WAN22_CKPT_DIR", DEFAULT_CKPT_DIR)))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame-num", type=int, default=121)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--sample-guide-scale", type=float, default=5.0)
    parser.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--offload-model", type=str_to_bool, default=True)
    parser.add_argument("--t5-cpu", type=str_to_bool, default=True)
    parser.add_argument("--convert-model-dtype", type=str_to_bool, default=True)
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
        "schema_version": "worldfoundry-wan22-runner-demo",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner": "worldfoundry.pipelines.wan.Wan2p2Pipeline",
        "official_reference": "Wan-Video/Wan2.2 generate.py ti2v-5B",
        "prompt": args.prompt,
        "task": "ti2v-5B",
        "size": args.size,
        "frame_num": args.frame_num,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "sample_solver": args.sample_solver,
        "offload_model": args.offload_model,
        "t5_cpu": args.t5_cpu,
        "convert_model_dtype": args.convert_model_dtype,
        "ckpt_dir": str(args.ckpt_dir),
        "output": str(output_path),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.ckpt_dir.is_dir():
        print(f"missing Wan2.2 TI2V-5B checkpoint directory: {args.ckpt_dir}", file=sys.stderr)
        return 4

    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    pipeline_module = importlib.import_module("worldfoundry.pipelines.wan.pipeline_wan_2p2")
    configs = importlib.import_module("worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.configs")
    utils = importlib.import_module("worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.utils.utils")
    wan_configs = configs.WAN_CONFIGS
    supported_sizes = configs.SUPPORTED_SIZES
    save_video = utils.save_video
    wan2p2_pipeline = pipeline_module.Wan2p2Pipeline

    if args.size not in supported_sizes["ti2v-5B"]:
        print(
            f"unsupported size {args.size!r}; supported: {', '.join(supported_sizes['ti2v-5B'])}",
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

    logging.info("Creating WorldFoundry Wan2.2 TI2V-5B runner pipeline.")
    pipeline = wan2p2_pipeline.from_pretrained(
        model_path=str(args.ckpt_dir),
        mode="ti2v-5B",
        device=local_rank,
        rank=rank,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("Generating Wan2.2 TI2V-5B runner parity video.")
    video = pipeline(
        prompt=args.prompt,
        images=None,
        size=args.size,
        frame_num=args.frame_num,
        sample_solver=args.sample_solver,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        base_seed=args.base_seed,
        offload_model=args.offload_model,
    )

    if rank == 0:
        logging.info("Saving generated video to %s", output_path)
        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=wan_configs["ti2v-5B"].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if not output_path.is_file():
            print(f"failed to save generated video: {output_path}", file=sys.stderr)
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
