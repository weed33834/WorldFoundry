from __future__ import annotations

from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "generate.py"
BLOCKED_REASON = (
    "Oasis 500M official source is vendored in-tree; execution still requires "
    "the official dependency environment, checkpoints, and task assets."
)


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    checks = {
        "oasis_ckpt": _option(options, "oasis_ckpt", "checkpoint_path", "model_path"),
        "vae_ckpt": _option(options, "vae_ckpt", "vae_checkpoint_path"),
        "prompt_path": _option(options, "prompt_path", "input_path", "image_path", "video_path"),
        "actions_path": _option(options, "actions_path", "action_path"),
    }
    missing = []
    for key, value in checks.items():
        if not value:
            missing.append({"kind": "option", "path": key, "reason": f"required Open-Oasis option `{key}` is not set"})
        elif not Path(value).expanduser().exists():
            missing.append({"kind": "asset", "path": value, "reason": f"Open-Oasis `{key}` path does not exist"})
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "Open-Oasis generate.py is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    return [
        context["python"],
        context["entrypoint"],
        "--oasis-ckpt",
        _option(options, "oasis_ckpt", "checkpoint_path", "model_path", default="") or "",
        "--vae-ckpt",
        _option(options, "vae_ckpt", "vae_checkpoint_path", default="") or "",
        "--prompt-path",
        _option(options, "prompt_path", "input_path", "image_path", "video_path", default="") or "",
        "--actions-path",
        _option(options, "actions_path", "action_path", default="") or "",
        "--output-path",
        context["output_path"],
        "--num-frames",
        str(options.get("num_frames", 32)),
        "--fps",
        str(options.get("fps", 20)),
        "--ddim-steps",
        str(options.get("ddim_steps", 10)),
    ]


__all__ = ["BLOCKED_REASON", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
