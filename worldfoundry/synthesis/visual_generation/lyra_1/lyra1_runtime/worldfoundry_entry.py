from __future__ import annotations

import os
import runpy
import sys


def _restrict_visible_cuda_devices_for_torchrun() -> None:
    if os.environ.get("WORLDFOUNDRY_LYRA1_PER_RANK_CUDA_VISIBLE", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text is None:
        return
    try:
        local_rank = int(local_rank_text)
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or "1")
    except ValueError:
        return
    if local_world_size <= 1:
        return

    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible_devices and local_rank < len(visible_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["WORLDFOUNDRY_LYRA1_LOCAL_CUDA_DEVICE"] = "0"
    os.environ.setdefault("WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE", "0")


_restrict_visible_cuda_devices_for_torchrun()

from worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c import (
    cosmos_predict1,
)
from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime import src


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--worldfoundry-script-path":
        raise SystemExit("usage: worldfoundry_entry.py --worldfoundry-script-path SCRIPT [SCRIPT_ARGS...]")

    script_path = sys.argv[2]
    sys.modules.setdefault("cosmos_predict1", cosmos_predict1)
    sys.modules.setdefault("src", src)
    sys.argv = [script_path, *sys.argv[3:]]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
