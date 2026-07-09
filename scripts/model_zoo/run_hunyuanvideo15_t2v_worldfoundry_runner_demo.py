#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_PROMPT = "A cat walks on a snowy street, cinematic, high quality."
DEFAULT_OUTPUT_NAME = "hunyuanvideo15_t2v_worldfoundry_seed42_848x480_9f.mp4"


def default_ckpt_dir() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", str(Path.home() / ".cache" / "worldfoundry" / "checkpoints"))) / "HunyuanVideo-1.5"


def default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR",
            str(REPO_ROOT / "tmp" / "model_zoo" / "parity" / "hunyuanvideo-1.5"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WorldFoundry HunyuanVideo-1.5 T2V runner demo used for model-zoo parity checks."
    )
    parser.add_argument("--ckpt-dir", type=Path, default=default_ckpt_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--video-length", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser


def write_metadata(path: Path, *, args: argparse.Namespace, output_path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "worldfoundry-hunyuanvideo15-runner-demo",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner": "worldfoundry.synthesis.visual_generation.official_video_runtime.OfficialVideoRuntime",
        "official_reference": "Tencent-Hunyuan/HunyuanVideo-1.5 generate.py t2v distilled",
        "prompt": args.prompt,
        "resolution": args.resolution,
        "aspect_ratio": args.aspect_ratio,
        "video_length": args.video_length,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "nproc_per_node": args.nproc_per_node,
        "ckpt_dir": str(args.ckpt_dir),
        "output": str(output_path),
        "result": result,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.ckpt_dir.is_dir():
        print(f"missing HunyuanVideo-1.5 checkpoint directory: {args.ckpt_dir}", file=sys.stderr)
        return 4

    runtime_mod = importlib.import_module("worldfoundry.synthesis.visual_generation.official_video_runtime")
    runtime = runtime_mod.OfficialVideoRuntime.from_model_id(
        "hunyuanvideo-1.5-t2v",
        device="cuda",
        checkpoint_path=str(args.ckpt_dir),
    )
    output_path = args.save_file or (args.output_dir / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = runtime.generate(
        prompt=args.prompt,
        output_path=output_path,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        video_length=args.video_length,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
        rewrite=False,
        cfg_distilled=True,
        enable_step_distill=False,
        sparse_attn=False,
        use_sageattn=False,
        enable_cache=False,
        sr=False,
        save_pre_sr_video=False,
        overlap_group_offloading=False,
    )
    metadata_path = args.metadata_json or output_path.with_suffix(".json")
    write_metadata(metadata_path, args=args, output_path=output_path, result=result)
    if result.get("status") != "succeeded" or not output_path.is_file():
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
