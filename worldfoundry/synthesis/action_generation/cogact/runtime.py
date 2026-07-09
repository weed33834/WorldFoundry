from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    ensure_import_path,
    first_present,
    import_from_workdir,
    load_image,
    option_bool,
    option_float,
    option_int,
    project_root,
    resolve_source_workdir,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_IN_TREE_RUNTIME = "worldfoundry/synthesis/action_generation/cogact/cogact_runtime"
_PRISMATIC_RUNTIME = "worldfoundry/synthesis/action_generation/openvla_oft/openvla_oft_runtime"


def _ensure_prismatic_runtime() -> None:
    runtime_root = project_root() / _PRISMATIC_RUNTIME
    if not (runtime_root / "prismatic").is_dir():
        raise FileNotFoundError(f"CogACT requires the in-tree Prismatic runtime package at {runtime_root / 'prismatic'}")
    ensure_import_path(runtime_root)


def _import_cogact_vla(workdir: Path) -> Any:
    workdir = workdir.expanduser().resolve()
    for name, module in list(sys.modules.items()):
        if name != "vla" and not name.startswith("vla."):
            continue
        module_file = Path(str(getattr(module, "__file__", ""))).expanduser()
        if not str(module_file).startswith(str(workdir)):
            sys.modules.pop(name, None)
    workdir_text = str(workdir)
    while workdir_text in sys.path:
        sys.path.remove(workdir_text)
    sys.path.insert(0, workdir_text)
    importlib.invalidate_caches()
    module = import_from_workdir("vla", workdir)
    if not hasattr(module, "load_vla"):
        raise ImportError(f"CogACT expected vla.load_vla from {workdir}, got {getattr(module, '__file__', None)}")
    return module


def _checkpoint_location(location: str) -> str:
    path = Path(str(location)).expanduser()
    if not path.exists() or path.is_file():
        return str(location)
    checkpoint_dir = path / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        preferred = [item for item in candidates if item.name.lower().startswith(path.name.split("--")[-1].lower())]
        if len(preferred) == 1:
            return str(preferred[0])
    raise FileNotFoundError(
        f"CogACT checkpoint directory must contain exactly one checkpoints/*.pt file, got {len(candidates)} in {checkpoint_dir}"
    )


def _hf_token(options: Mapping[str, Any]) -> str | None:
    token = options.get("hf_token") or options.get("huggingface_token")
    if token:
        return str(token)
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    location = _checkpoint_location(location)
    hf_token = _hf_token(options)
    workdir = resolve_source_workdir(
        options,
        "cogact",
        specific_env="WORLDFOUNDRY_COGACT_REPO",
        in_tree_subdir=_IN_TREE_RUNTIME,
    )
    key = (
        str(workdir),
        location,
        device,
        options.get("cache_dir"),
        hf_token,
        options.get("action_model_type"),
        options.get("future_action_window_size"),
        options.get("past_action_window_size"),
        options.get("use_ema"),
    )
    model = _RUNTIME_CACHE.get(key)
    if model is not None:
        return model

    _ensure_prismatic_runtime()
    vla_module = _import_cogact_vla(workdir)
    model = vla_module.load_vla(
        location,
        hf_token=hf_token,
        cache_dir=options.get("cache_dir"),
        load_for_training=False,
        action_model_type=str(options.get("action_model_type") or "DiT-B"),
        future_action_window_size=option_int(options.get("future_action_window_size"), 15),
        past_action_window_size=option_int(options.get("past_action_window_size"), 0),
        use_ema=option_bool(options.get("use_ema"), False),
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    _RUNTIME_CACHE[key] = model
    return model


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del action_context
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "CogACT/CogACT-Base")
    policy = _runtime_for(location, device, options)
    selected = first_present(observation, "image", "full_image")
    if selected is None:
        selected = image
    selected_image = load_image(selected)
    if selected_image is None:
        raise ValueError("CogACT requires an image/full_image observation for direct inference.")
    raw_actions = policy.predict_action(
        selected_image,
        instruction,
        unnorm_key=str(first_present(observation, "unnorm_key") or options.get("unnorm_key") or "bridge_orig"),
        cfg_scale=option_float(first_present(observation, "cfg_scale") or options.get("cfg_scale"), 1.5),
        use_ddim=option_bool(first_present(observation, "use_ddim") or options.get("use_ddim"), False),
        num_ddim_steps=option_int(first_present(observation, "num_ddim_steps") or options.get("num_ddim_steps"), 5),
    )
    actions = raw_actions[0] if isinstance(raw_actions, tuple) else raw_actions
    return completed_action_result(
        model_id="cogact",
        instruction=instruction,
        actions=actions,
        raw_output=raw_actions,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.cogact.native_in_process",
        metadata={
            "official_entrypoint": "vla:load_vla(...).predict_action",
            "source_workdir": str(
                resolve_source_workdir(
                    options,
                    "cogact",
                    specific_env="WORLDFOUNDRY_COGACT_REPO",
                    in_tree_subdir=_IN_TREE_RUNTIME,
                )
            ),
        },
    )
