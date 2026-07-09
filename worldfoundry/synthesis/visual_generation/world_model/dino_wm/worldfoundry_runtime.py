from __future__ import annotations

import importlib.util
from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
DEFAULT_CONFIG = resolve_data_path("models", "runtime", "configs", "dino_wm", "conf", "plan_wall.yaml")
BLOCKED_REASON = (
    "DINO-WM official planning/evaluation route is vendored in-tree; execution requires "
    "official checkpoints and environment/task assets."
)


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    config = Path(str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG)).expanduser()
    ckpt_base_path = options.get("ckpt_base_path") or options.get("checkpoint_dir") or options.get("checkpoint_path")
    model_name = options.get("model_name") or options.get("run_name")
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "DINO-WM infer.py is missing"})
    if not config.is_file():
        missing.append({"kind": "asset", "path": str(config), "reason": "DINO-WM planning config does not exist"})
    for module_name in ("gym", "hydra", "omegaconf", "einops"):
        if importlib.util.find_spec(module_name) is None:
            missing.append({"kind": "python_module", "path": module_name, "reason": "required DINO-WM runtime package is not importable"})
    if not ckpt_base_path:
        missing.append({"kind": "checkpoint", "path": "ckpt_base_path", "reason": "DINO-WM requires ckpt_base_path/checkpoint_dir"})
    elif not Path(str(ckpt_base_path)).expanduser().exists():
        missing.append({"kind": "checkpoint", "path": str(ckpt_base_path), "reason": "DINO-WM checkpoint base path does not exist"})
    if not model_name:
        missing.append({"kind": "option", "path": "model_name", "reason": "DINO-WM requires model_name/run_name"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    ckpt_base_path = options.get("ckpt_base_path") or options.get("checkpoint_dir") or options.get("checkpoint_path") or ""
    model_name = options.get("model_name") or options.get("run_name") or ""
    command = [
        context["python"],
        context["entrypoint"],
        "--config",
        str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG),
        "--ckpt-base-path",
        str(ckpt_base_path),
        "--model-name",
        str(model_name),
        "--model-epoch",
        str(options.get("model_epoch", "latest")),
        "--output-dir",
        context["output_dir"],
    ]
    for option_key, flag in (
        ("seed", "--seed"),
        ("n_evals", "--n-evals"),
        ("goal_source", "--goal-source"),
        ("goal_H", "--goal-h"),
    ):
        if option_key in options and options[option_key] not in (None, ""):
            command.extend([flag, str(options[option_key])])
    return command


__all__ = ["BLOCKED_REASON", "DEFAULT_CONFIG", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
