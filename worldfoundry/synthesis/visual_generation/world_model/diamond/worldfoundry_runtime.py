from __future__ import annotations

import importlib.util
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path, resolve_data_path

RUNTIME_DIR = Path(__file__).resolve().parent / "diamond_runtime"
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "play.py"
CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "diamond", "config")
DEFAULT_PRETRAINED_DIR = checkpoint_root_path("diamond")
BLOCKED_REASON = (
    "DIAMOND official play/inference source is vendored in-tree; execution requires "
    "a local checkpoint or non-interactive pretrained HF download plus an Atari runtime."
)


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    config_dir = Path(str(options.get("config_dir") or CONFIG_DIR)).expanduser()
    checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
    pretrained = bool(options.get("pretrained", False))
    pretrained_game = str(options.get("pretrained_game") or options.get("game") or "Pong")
    pretrained_dir = Path(str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR)).expanduser()
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "DIAMOND play.py is missing"})
    if not config_dir.is_dir() or not (config_dir / "trainer.yaml").is_file():
        missing.append({"kind": "asset", "path": str(config_dir), "reason": "DIAMOND runtime config directory is missing trainer.yaml"})
    for module_name in (
        "huggingface_hub",
        "hydra",
        "omegaconf",
        "pygame",
        "gymnasium",
        "ale_py",
        "cv2",
        "PIL",
        "torcheval",
        "tqdm",
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append({"kind": "python_module", "path": module_name, "reason": "required DIAMOND runtime package is not importable"})
    if not pretrained:
        if not checkpoint:
            missing.append(
                {
                    "kind": "checkpoint",
                    "path": "checkpoint",
                    "reason": "DIAMOND requires checkpoint/checkpoint_path/model_path unless pretrained=true",
                }
            )
        elif not Path(str(checkpoint)).expanduser().is_file():
            missing.append({"kind": "checkpoint", "path": str(checkpoint), "reason": "DIAMOND checkpoint file does not exist"})
    elif "pretrained_dir" in options or pretrained_dir.is_dir():
        for relative in (
            f"atari_100k/models/{pretrained_game}.pt",
            "atari_100k/config/agent/default.yaml",
            "atari_100k/config/env/atari.yaml",
        ):
            path = pretrained_dir / relative
            if not path.is_file():
                missing.append({"kind": "checkpoint", "path": str(path), "reason": "DIAMOND local pretrained asset is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    command = [
        context["python"],
        context["entrypoint"],
        "--config-dir",
        str(options.get("config_dir") or CONFIG_DIR),
        "--fps",
        str(options.get("fps", 15)),
        "--size",
        str(options.get("size", 640)),
        "--num-steps-initial-collect",
        str(options.get("num_steps_initial_collect", 1000)),
        "--no-header",
    ]
    checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
    if bool(options.get("pretrained", False)):
        command.append("--pretrained")
        game = options.get("pretrained_game") or options.get("game") or "Pong"
        command.extend(["--pretrained-game", str(game)])
        pretrained_dir = Path(str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR)).expanduser()
        if pretrained_dir.is_dir() or options.get("pretrained_dir"):
            command.extend(["--pretrained-dir", str(pretrained_dir)])
    elif checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if bool(options.get("record", False)):
        command.append("--record")
    if bool(options.get("store_denoising_trajectory", False)):
        command.append("--store-denoising-trajectory")
    if bool(options.get("store_original_obs", False)):
        command.append("--store-original-obs")
    return command


__all__ = [
    "BLOCKED_REASON",
    "CONFIG_DIR",
    "DEFAULT_PRETRAINED_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
