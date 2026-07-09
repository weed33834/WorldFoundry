from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    import_from_workdir,
    load_json_if_present,
    option_bool,
    option_float,
    option_int,
    resolve_source_workdir,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_IN_TREE_RUNTIME = "worldfoundry/synthesis/action_generation/db_cogact/db_cogact_runtime"


def _normalization_stats(checkpoint_path: str, options: Mapping[str, Any]) -> dict[str, Any]:
    explicit = first_present(options, "norm_stats", "norm_stats_path", "normalization_stats_path")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if explicit is not None:
        loaded = load_json_if_present(str(explicit))
        if loaded is not None:
            if "norm_stats" in loaded:
                loaded = loaded["norm_stats"]
            return dict(loaded.get("default", loaded))
    loaded = load_json_if_present(Path(checkpoint_path) / "norm_stats.json")
    if loaded is not None:
        if "norm_stats" in loaded:
            loaded = loaded["norm_stats"]
        return dict(loaded.get("default", loaded))
    return {"min": -1, "max": 1}


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    workdir = resolve_source_workdir(
        options,
        "db-cogact",
        specific_env="WORLDFOUNDRY_DB_COGACT_REPO",
        in_tree_subdir=_IN_TREE_RUNTIME,
    )
    camera_order = tuple(str(item) for item in options.get("camera_order") or ("front", "left_wrist", "right_wrist"))
    key = (str(workdir), location, device, camera_order, options.get("torch_dtype"))
    policy = _RUNTIME_CACHE.get(key)
    if policy is not None:
        return policy

    cogact_arch = import_from_workdir("dexbotic.model.cogact.cogact_arch", workdir)
    policy_module = import_from_workdir("dexbotic.policy.cogact_policy", workdir)
    transformers = import_from_workdir("transformers", workdir)
    import torch

    dtype = torch.bfloat16 if str(options.get("torch_dtype") or "bfloat16").lower() in {"bf16", "bfloat16"} else torch.float32
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": option_bool(options.get("trust_remote_code"), True),
    }
    device_text = str(device or "cpu")
    requested_cuda = device_text.startswith("cuda")
    if requested_cuda and torch.cuda.is_available():
        load_kwargs["device_map"] = {"": "cuda:0"}
    model = cogact_arch.CogACTForCausalLM.from_pretrained(location, **load_kwargs)
    if hasattr(model, "to"):
        target_device = "cpu" if requested_cuda and not torch.cuda.is_available() else device_text
        model = model.to(torch.device(target_device))
    if hasattr(model, "eval"):
        model = model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(location, trust_remote_code=option_bool(options.get("trust_remote_code"), True))
    policy = policy_module.CogACTPolicy(
        model=model,
        tokenizer=tokenizer,
        norm_stats=_normalization_stats(location, options),
        camera_order=list(camera_order),
    )
    _RUNTIME_CACHE[key] = policy
    return policy


def _policy_observation(observation: Mapping[str, Any], image: Any, instruction: str, options: Mapping[str, Any]) -> dict[str, Any]:
    camera_order = [str(item) for item in options.get("camera_order") or ("front", "left_wrist", "right_wrist")]
    payload: dict[str, Any] = {"prompt": first_present(observation, "prompt") or instruction}
    for key, value in observation.items():
        if key.startswith("image/") and value is not None:
            payload[key] = value
    if not any(key.startswith("image/") for key in payload):
        images = collect_images(observation, image, [*camera_order, "1", "2", "3"])
        for index, item in enumerate(images, start=1):
            payload[f"image/{index}"] = item
    state = first_present(observation, "state", "robot_state", "proprio")
    if state is not None:
        payload["state"] = state
    return payload


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
    location = checkpoint_path or str(options.get("checkpoint_ref") or "Dexmal/libero-db-cogact")
    policy = _runtime_for(location, device, options)
    policy_obs = _policy_observation(observation, image, instruction, options)
    if not any(key.startswith("image/") for key in policy_obs):
        raise ValueError("DB-CogACT requires at least one image slot for direct select_action inference.")
    types_module = import_from_workdir(
        "dexbotic.policy.types",
        resolve_source_workdir(
            options,
            "db-cogact",
            specific_env="WORLDFOUNDRY_DB_COGACT_REPO",
            in_tree_subdir=_IN_TREE_RUNTIME,
        ),
    )
    sampling = types_module.SamplingConfig(
        num_steps=option_int(options.get("num_steps"), 10),
        cfg_scale=option_float(options.get("cfg_scale"), 1.5),
        seed=options.get("seed"),
    )
    raw = policy.select_action(policy_obs, sampling)
    return completed_action_result(
        model_id="db-cogact",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.db_cogact.native_in_process",
        metadata={
            "official_entrypoint": "dexbotic.policy.base_policy:BasePolicy.select_action",
            "action_mode": getattr(policy, "action_mode", None),
            "state_used": getattr(policy, "state_used", None),
        },
    )
