from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    import_from_workdir,
    option_int,
    resolve_source_workdir,
    to_numpy_image,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
_IN_TREE_RUNTIME = "worldfoundry/synthesis/action_generation/vlanext/vlanext_runtime"


def _cfg(location: str, options: Mapping[str, Any]) -> Any:
    model_options = {}
    if options.get("diffusion_steps") is not None:
        model_options["diffusion_steps"] = option_int(options.get("diffusion_steps"), 10)
    if options.get("scheduler_type") is not None:
        model_options["scheduler_type"] = str(options["scheduler_type"])
    return SimpleNamespace(eval=SimpleNamespace(finetuned_checkpoint=location), model=SimpleNamespace(**model_options))


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    workdir = resolve_source_workdir(
        options,
        "vlanext",
        specific_env="WORLDFOUNDRY_VLANEXT_REPO",
        in_tree_subdir=_IN_TREE_RUNTIME,
    )
    key = (
        str(workdir),
        location,
        device,
        options.get("diffusion_steps"),
        options.get("scheduler_type"),
    )
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached
    utils = import_from_workdir("src.evaluation.libero_bench.VLANeXt_utils", workdir)
    cfg = _cfg(location, options)
    model = utils.get_vla(cfg)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    processor = utils.get_processor(cfg)
    cached = (cfg, model, processor)
    _RUNTIME_CACHE[key] = cached
    return cached


def _observation(observation: Mapping[str, Any], image: Any, instruction: str) -> dict[str, Any]:
    full_image = first_present(observation, "full_image", "image")
    if full_image is None and isinstance(image, Mapping):
        full_image = first_present(image, "full_image", "image")
    if full_image is None:
        full_image = image
    wrist = first_present(observation, "full_image_wrist", "wrist_image")
    if wrist is None and isinstance(image, Mapping):
        wrist = first_present(image, "full_image_wrist", "wrist_image")

    obs = dict(observation)
    obs["full_image"] = to_numpy_image(full_image)
    if wrist is not None:
        obs["full_image_wrist"] = to_numpy_image(wrist)
    if obs.get("image_history") is not None:
        obs["image_history"] = [to_numpy_image(item) for item in obs["image_history"]]
    if obs.get("image_history_wrist") is not None:
        obs["image_history_wrist"] = [to_numpy_image(item) for item in obs["image_history_wrist"]]
    obs.setdefault("task_description", instruction)
    return obs


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
    location = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not location:
        raise ValueError("VLANeXt requires checkpoint_path pointing to a VLANeXt_*.pt checkpoint.")
    cfg, model, processor = _runtime_for(location, device, options)
    obs = _observation(observation, image, instruction)
    if obs.get("full_image") is None:
        raise ValueError("VLANeXt requires full_image or image input for direct inference.")
    raw = import_from_workdir(
        "src.evaluation.libero_bench.VLANeXt_utils",
        resolve_source_workdir(
            options,
            "vlanext",
            specific_env="WORLDFOUNDRY_VLANEXT_REPO",
            in_tree_subdir=_IN_TREE_RUNTIME,
        ),
    ).get_vla_action(cfg, model, processor, obs, instruction)
    return completed_action_result(
        model_id="vlanext",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.vlanext.native_in_process",
        metadata={
            "official_entrypoint": "src.evaluation.libero_bench.VLANeXt_utils:get_vla_action",
            "task_suite_name": options.get("task_suite_name"),
        },
    )
