from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    ensure_import_path,
    first_present,
    import_from_workdir,
    option_int,
    resolve_source_workdir,
    to_numpy_image,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_IN_TREE_RUNTIME = "worldfoundry/synthesis/action_generation/mme_vla/mme_vla_runtime"
_SHARED_OPENPI_RUNTIME = Path(__file__).resolve().parents[1] / "openpi" / "openpi_runtime"


def _runtime_for(location: str, options: Mapping[str, Any]) -> Any:
    workdir = resolve_source_workdir(
        options,
        "mme-vla",
        specific_env="WORLDFOUNDRY_MME_VLA_REPO",
        default_subdir="src",
        in_tree_subdir=_IN_TREE_RUNTIME,
    )
    config_name = str(options.get("policy_config") or "mme_vla_suite")
    key = (
        str(workdir),
        location,
        config_name,
        options.get("seed"),
        options.get("default_prompt"),
    )
    policy = _RUNTIME_CACHE.get(key)
    if policy is not None:
        return policy

    ensure_import_path(_SHARED_OPENPI_RUNTIME)
    config_module = import_from_workdir("mme_vla_suite.training.config", workdir)
    policy_config = import_from_workdir("mme_vla_suite.policies.policy_config", workdir)
    train_config = config_module.get_config(config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        Path(location),
        seed=option_int(options.get("seed"), 42),
        default_prompt=options.get("default_prompt"),
    )
    _RUNTIME_CACHE[key] = policy
    return policy


def _policy_observation(observation: Mapping[str, Any], image: Any, instruction: str) -> dict[str, Any]:
    base_image = first_present(observation, "observation/image", "image", "base_image")
    wrist = first_present(observation, "observation/wrist_image", "wrist_image")
    if base_image is None and isinstance(image, Mapping):
        base_image = first_present(image, "observation/image", "image", "base_image")
        wrist = wrist if wrist is not None else first_present(image, "observation/wrist_image", "wrist_image")
    elif base_image is None:
        base_image = image

    payload: dict[str, Any] = {
        "observation/image": to_numpy_image(base_image),
        "observation/wrist_image": to_numpy_image(wrist),
        "observation/state": first_present(observation, "observation/state", "state", "robot_state", "proprio"),
        "prompt": first_present(observation, "prompt", "task_instruction", "instruction") or instruction,
    }
    for key in (
        "static_image_emb",
        "static_pos_emb",
        "static_state_emb",
        "static_mask",
        "recur_image_emb",
        "recur_pos_emb",
        "recur_state_emb",
        "recur_mask",
        "simple_subgoal",
        "grounded_subgoal",
    ):
        if key in observation and observation[key] is not None:
            payload[key] = observation[key]
    return payload


def _buffer_observation(observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    state = first_present(observation, "state", "states", "observation/state", "robot_state", "proprio")
    images = first_present(observation, "images")
    if images is not None and state is not None:
        return {
            "images": images,
            "state": state,
            "exec_start_idx": option_int(observation.get("exec_start_idx"), 0),
        }

    image = first_present(observation, "observation/image", "image", "base_image")
    image_np = to_numpy_image(image)
    if image_np is None or state is None:
        raise ValueError("MME-VLA history_observations require images/state or observation/image plus observation/state.")

    import numpy as np

    return {
        "images": np.asarray(image_np, dtype=np.uint8)[None, None, ...],
        "state": np.asarray(state, dtype=np.float32)[None, ...],
        "exec_start_idx": option_int(observation.get("exec_start_idx"), 0),
        "prompt": first_present(observation, "prompt", "task_instruction", "instruction") or instruction,
    }


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
    del action_context, device
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not location:
        raise ValueError("MME-VLA requires checkpoint_path pointing to a policy checkpoint step directory.")
    policy = _runtime_for(location, options)
    histories = observation.get("history_observations") or ()
    if isinstance(histories, Mapping):
        histories = (histories,)
    for history in histories:
        policy.add_buffer(_buffer_observation(history, instruction))
    obs = _policy_observation(observation, image, instruction)
    if obs.get("observation/image") is None or obs.get("observation/wrist_image") is None:
        raise ValueError("MME-VLA requires observation/image and observation/wrist_image inputs.")
    if obs.get("observation/state") is None:
        raise ValueError("MME-VLA requires observation/state input.")
    raw = policy.infer(obs)
    return completed_action_result(
        model_id="mme-vla",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device="jax",
        runtime="worldfoundry.mme_vla.native_in_process",
        metadata={
            "official_entrypoint": "mme_vla_suite.policies.policy_config:create_trained_policy(...).infer",
            "policy_config": options.get("policy_config") or "mme_vla_suite",
        },
    )
