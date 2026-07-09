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

DEFAULT_PROMPT = "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick."
DEFAULT_OUTPUT_NAME = "hunyuanvideo_i2v_worldfoundry_firework_864x1024_129f.mp4"


def default_ckpt_dir() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", str(Path.home() / ".cache" / "worldfoundry" / "checkpoints"))) / "HunyuanVideo-I2V"


def default_image_path() -> Path:
    return REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "hunyuanvideo_i2v" / "0.jpg"


def default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR",
            str(REPO_ROOT / "tmp" / "model_zoo" / "parity" / "hunyuanvideo-i2v"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WorldFoundry HunyuanVideo I2V runner demo used for model-zoo parity checks."
    )
    parser.add_argument("--ckpt-dir", type=Path, default=default_ckpt_dir())
    parser.add_argument("--image-path", type=Path, default=default_image_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--ulysses-degree", type=int, default=8)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--flow-shift", type=float, default=7.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--i2v-resolution", default="720p")
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser


def write_metadata(path: Path, *, args: argparse.Namespace, output_path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "worldfoundry-hunyuanvideo-i2v-runner-demo",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner": "worldfoundry.synthesis.visual_generation.official_video_runtime.OfficialVideoRuntime",
        "official_reference": "Tencent-Hunyuan/HunyuanVideo-I2V sample_image2video.py stability demo",
        "prompt": args.prompt,
        "image_path": str(args.image_path),
        "i2v_resolution": args.i2v_resolution,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "nproc_per_node": args.nproc_per_node,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "ckpt_dir": str(args.ckpt_dir),
        "output": str(output_path),
        "result": result,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.ckpt_dir.is_dir():
        print(f"missing HunyuanVideo-I2V checkpoint directory: {args.ckpt_dir}", file=sys.stderr)
        return 4
    if not args.image_path.is_file():
        print(f"missing HunyuanVideo-I2V demo image: {args.image_path}", file=sys.stderr)
        return 5

    runtime_mod = importlib.import_module("worldfoundry.synthesis.visual_generation.official_video_runtime")
    runtime = runtime_mod.OfficialVideoRuntime.from_model_id(
        "hunyuanvideo-i2v",
        device="cuda",
        checkpoint_path=str(args.ckpt_dir),
    )
    output_path = args.save_file or (args.output_dir / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = runtime.generate(
        prompt=args.prompt,
        image_path=args.image_path,
        output_path=output_path,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        flow_shift=args.flow_shift,
        embedded_cfg_scale=args.embedded_cfg_scale,
        i2v_resolution=args.i2v_resolution,
        i2v_stability=True,
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
