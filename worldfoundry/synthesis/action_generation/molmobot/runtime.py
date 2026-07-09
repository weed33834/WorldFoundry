from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    import_from_workdir,
    option_bool,
    option_int,
    resolve_source_workdir,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_IN_TREE_RUNTIME = "worldfoundry/synthesis/action_generation/molmobot/molmobot_runtime"


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    variant = str(options.get("variant") or options.get("runtime_variant") or "").lower()
    if "pi0" in variant:
        raise RuntimeError(
            "MolmoBot-Pi0 direct policy construction depends on OpenPI and molmo_spaces extras; "
            "the native WorldFoundry runtime currently supports the OLMo SynthManip MolmoBot path."
        )
    workdir = resolve_source_workdir(
        options,
        "molmobot",
        specific_env="WORLDFOUNDRY_MOLMOBOT_REPO",
        default_subdir="MolmoBot",
        in_tree_subdir=_IN_TREE_RUNTIME,
    )
    key = (
        str(workdir),
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

    module = import_from_workdir("olmo.models.molmobot.inference_wrapper", workdir)
    agent = module.SynthManipMolmoInferenceWrapper(
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
            "official_entrypoint": "olmo.models.molmobot.inference_wrapper:SynthManipMolmoInferenceWrapper.get_action_chunk",
            "camera_keys": list(camera_keys),
            "agent_config": getattr(agent, "config", {}),
        },
    )
