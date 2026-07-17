"""Internal torchrun worker for one Bernini inference request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .worldfoundry_runtime import BerniniRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    runtime = BerniniRuntime(
        model_id=payload["model_id"],
        checkpoint_path=payload["checkpoint_path"],
        device="cuda",
        model_config=payload.get("model_config"),
    )
    try:
        runtime.generate(
            prompt=payload["prompt"],
            output_path=payload["output_path"],
            task_type=payload["task_type"],
            image=payload.get("image"),
            images=payload.get("images"),
            video=payload.get("video"),
            ulysses_size=int(payload["ulysses_size"]),
            **payload.get("overrides", {}),
        )
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
