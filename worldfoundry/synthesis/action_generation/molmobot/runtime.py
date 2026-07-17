from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}


def clear_runtime_cache() -> None:
    from worldfoundry.core.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    variant = str(options.get("variant") or options.get("runtime_variant") or "").lower()
    if "pi0" in variant:
        raise RuntimeError(
            "MolmoBot-Pi0 direct policy construction depends on OpenPI and molmo_spaces extras; "
            "the native WorldFoundry runtime currently supports the OLMo SynthManip MolmoBot path."
        )
    key = (
        location,
        device,
        options.get("num_flow_steps"),
        options.get("max_seq_len"),
        options.get("norm_repo_id"),
        options.get("states_mode"),
        options.get("use_bfloat16"),
        options.get("compile_model"),
    )
    agent = _RUNTIME_CACHE.get(key)
    if agent is not None:
        return agent

    from .policy import SynthManipMolmoInferenceWrapper

    agent = SynthManipMolmoInferenceWrapper(
        checkpoint_path=location,
        device=device,
        num_flow_steps=None if options.get("num_flow_steps") in (None, "") else option_int(options.get("num_flow_steps"), 10),
        max_seq_len=None if options.get("max_seq_len") in (None, "") else option_int(options.get("max_seq_len"), 0),
        norm_repo_id=str(options.get("norm_repo_id") or "synthmanip"),
        use_bfloat16=option_bool(options.get("use_bfloat16"), True),
        compile_model=option_bool(options.get("compile_model"), False),
        states_mode=options.get("states_mode"),
    )
    _RUNTIME_CACHE[key] = agent
    return agent


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
    location = checkpoint_path or str(options.get("checkpoint_ref") or "allenai/MolmoBot-DROID")
    agent = _runtime_for(location, device, options)
    camera_keys = tuple(str(item) for item in options.get("camera_keys") or observation.get("camera_keys") or ("exo_camera_1", "wrist_camera"))
    images = collect_images(observation, image, camera_keys)
    if not images:
        raise ValueError("MolmoBot requires one or more camera images for get_action_chunk.")
    state = first_present(observation, "state", "qpos", "robot_state", "joint_state")
    raw = agent.get_action_chunk(
        images=images,
        task_description=str(first_present(observation, "task", "task_description") or instruction),
        state=state,
    )
    return completed_action_result(
        model_id="molmobot",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.molmobot.native_in_process",
        metadata={
            "entrypoint": "worldfoundry.synthesis.action_generation.molmobot.policy:SynthManipMolmoInferenceWrapper.get_action_chunk",
            "camera_keys": list(camera_keys),
            "agent_config": getattr(agent, "config", {}),
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
